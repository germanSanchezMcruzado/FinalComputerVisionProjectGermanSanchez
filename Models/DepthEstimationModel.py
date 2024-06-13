import tensorflow as tf

from sklearn.model_selection import ParameterGrid
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input, Concatenate, UpSampling2D, Reshape, Lambda, Add
from keras.optimizers import Adam, SGD, RMSprop


class DepthEstimationModel:
    def __init__(self, input_shape_gaussian, input_shape_edge, input_shape_original,pretrained_path = ""):
        self.input_shape_gaussian = input_shape_gaussian
        self.input_shape_edge = input_shape_edge
        self.input_shape_original = input_shape_original

        if pretrained_path == "":
            self.model = self.build_model()
        else:
            self.model = self.load_model(pretrained_path)

    def build_gaussian_module(self):
        '''
        This module receives 4 gaussians with shape 4 * W * H * 3
        Treats each image with same kernel and returns feature image of size W * H * 1
        '''

        num_images, img_height, img_width, channels = self.input_shape_gaussian

        '''
        applies a conv2D to return image of size W * H * 1
        '''
        def gaussian_conv_kernel(x):
            output = Conv2D(16,(3, 3), padding='same', activation='relu')(x)
            output = Conv2D(8, (3, 3), padding='same', activation='relu')(output)
            output = Conv2D(1, (3, 3), padding='same', activation='relu')(output)
            return output

        input = Input(shape=(num_images, img_height, img_width, channels))

        # Apply shared convolutions to each of the 4 images
        processed_images = [gaussian_conv_kernel(Lambda(lambda z: z[:, i, :, :, :])(input)) for i in range(num_images)]

        # Concatenate the processed images along the channel dimension
        concatenated = Concatenate(axis=-1)(processed_images) #W*H*4

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(concatenated)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(4, 4))(x)
        x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

        outputs = Reshape((img_height, img_width, 1))(x)

        return Model(input, outputs, name='GaussianModule')

    def build_edge_module(self):
        '''
        This module receives inputs of 5 * W * H * 3 images.
        Applies different convolutions for each image and merges them into a single output layer with size W * H * 1.
        '''
        num_images, img_height, img_width, channels = self.input_shape_edge

        input = Input(shape=(num_images, img_height, img_width, channels))

        # Define the shared convolutional layers
        def edge_independent_conv_layers(x):
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            return x

        # Apply shared convolutions to each of the 5 images
        processed_images = [edge_independent_conv_layers(Lambda(lambda z: z[:, i, :, :, :])(input)) for i in range(num_images)]

        # Concatenate the processed images along the channel dimension
        concatenated = Concatenate(axis=-1)(processed_images) #W*H*5

        # Further processing to merge into a single output of size W * H * 1
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(concatenated)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(1, (3, 3), activation='linear', padding='same')(x)

        outputs = Reshape((img_height, img_width, 1))(x)  # Adjust the size according to pooling layers

        return Model(input, outputs, name='EdgeModule')

    def build_merge_module(self, gaussian_output_shape, edge_output_shape):
        '''
        Receives W * H * 5 (2 channels for Gaussian and edge, and 3 for original)
        Both are treated separately and then merged to one W * H * 1 depth output
        '''
        # Define the input shapes
        input_gaussian = Input(shape=gaussian_output_shape[1:])  # Shape: W * H * 1
        input_edge = Input(shape=edge_output_shape[1:])  # Shape: W * H * 1
        input_original = Input(shape=self.input_shape_original)  # Shape: W * H * 3

        # Concatenate Gaussian and Edge channels along the channel dimension
        combined_inputs = Concatenate(axis=-1)([input_gaussian, input_edge])  # Shape: W * H * 2
        # Process combined Gaussian and Edge channels
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(combined_inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        processed_combined = Conv2D(4, (3, 3), activation='relu', padding='same')(x)  # Shape: W/4 * H/4 * 128

        # Process original image channels
        y = Conv2D(16, (3, 3), activation='relu', padding='same')(input_original)
        y = MaxPooling2D((2, 2))(y)
        y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)
        y = MaxPooling2D((2, 2))(y)
        processed_original = Conv2D(4, (3, 3), activation='relu', padding='same')(y)  # Shape: W/4 * H/4 * 128

        # Merge the processed outputs
        merged = Add()([processed_combined, processed_original])  # Element-wise addition

        # Further processing of the merged output
        z = Conv2D(16, (3, 3), activation='relu', padding='same')(merged)
        z = Conv2D(8, (3, 3), activation='relu', padding='same')(z)
        z = UpSampling2D(size=(2,2))(z)
        z = Conv2D(4, (3, 3), activation='relu', padding='same')(z)
        z = UpSampling2D(size=(2, 2))(z)
        z = Conv2D(1, (3, 3), activation='linear', padding='same')(z)  # Final output layer W * H * 1

        return Model(inputs=[input_gaussian, input_edge, input_original], outputs=z, name='MergeModule')

    def build_model(self):
        gaussian_module = self.build_gaussian_module()
        edge_module = self.build_edge_module()

        gaussian_output_shape = gaussian_module.output_shape
        edge_output_shape = edge_module.output_shape

        merge_module = self.build_merge_module(gaussian_output_shape, edge_output_shape)

        input_gaussian = Input(shape=self.input_shape_gaussian)
        input_edge = Input(shape=self.input_shape_edge)
        input_original = Input(shape=self.input_shape_original)

        gaussian_output = gaussian_module(input_gaussian)
        edge_output = edge_module(input_edge)

        final_output = merge_module([gaussian_output, edge_output, input_original])

        model = Model([input_gaussian, input_edge, input_original], final_output)

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        return model

    def load_model(self, path):
        model = tf.keras.models.load_model(path)

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        return model

    def grid_search(self, pipeline, param_grid, verbose=True, save_best_model=False, save_path=None):
        """
           Custom grid search implementation using the traditional fit method with generators.

           Args:
           - model: Keras model object.
           - dataset_generator: Generator yielding (X, Y) tuples.
           - param_grid: Dictionary specifying the hyperparameter grid.
           - epochs: Number of epochs to train the model for each combination of hyperparameters.
           - verbose: Whether to print progress messages.
           - save_best_model: Whether to save the best model found during grid search.
           - save_path: Path to save the best model.

           Returns:
           - best_model: The best model found during grid search.
           - best_params: The best hyperparameters found.
           """

        best_model = None
        best_score = float('-inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            if verbose:
                print("Training with parameters:", params)

            # Train the model
            pipeline.generator = tf.data.Dataset.from_generator(output_signature=pipeline.output_signature, generator=pipeline.execute_pipeline_once, args=(params['batch_size'],))
            optimizer = None
            if params['optimizer'] == 'adam':
                optimizer = Adam(learning_rate=params['learning_rate'])
            elif params['optimizer'] == 'rmsprop':
                optimizer = RMSprop(learning_rate=params['learning_rate'])
            else:
                optimizer = SGD(learning_rate=params['learning_rate'])

            self.model.compile(optimizer=optimizer, loss=params['loss'])
            history = self.model.fit(pipeline.generator, epochs=params['epochs'])

            # Evaluate the model
            score = history.history['loss'][-1]

            if score > best_score:
                best_score = score
                best_params = params
                best_model = self.model

            if verbose:
                print("  - Score:", score)

        if verbose:
            print("Best parameters:", best_params)
            print("Best score:", best_score)

        # Save the best model if specified
        if save_best_model and best_model is not None and save_path is not None:
            best_model.save(save_path)
            if verbose:
                print("Best model saved to:", save_path)

        return best_model, best_params

    def evaluate(self, test_data, test_labels, verbose=True):
        if verbose:
            print("Evaluation started...")
        loss = self.model.evaluate(test_data, test_labels, verbose=verbose)
        if verbose:
            print("Evaluation completed.")
        return loss

    def predict(self, inputs, verbose=True):
        if verbose:
            print("Prediction started...")
        predictions = self.model.predict(inputs, verbose=verbose)
        if verbose:
            print("Prediction completed.")
        return predictions
