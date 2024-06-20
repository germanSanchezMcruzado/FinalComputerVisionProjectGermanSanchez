import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU, concatenate, UpSampling2D, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import ParameterGrid
from tensorflow.keras import backend as K

class DepthEstimationModel:
    def __init__(self, input_shape=None, saved_path="", init_lr=0.001, epochs=50):
        self.saved_path = saved_path
        self.input_shape = input_shape
        self.model = None
        self.INIT_LR = init_lr
        self.EPOCHS = epochs


        if saved_path != "":
            self.model = self.load_model()
        else:
            self.model = self.build_model()

        self.model.summary()

    def load_model(self):
        model = tf.keras.models.load_model(self.saved_path, custom_objects={
            'depth_loss': self.depth_loss,
            'depth_acc': self.depth_acc
        })

        return model

    def downsampling_block(self,input_tensor, n_filters):
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(input_tensor)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        return x

    def upsampling_block(self,input_tensor, n_filters, name, concat_with):
        x = UpSampling2D((2, 2), interpolation='bilinear', name=name)(input_tensor)
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name + "_convA")(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = concatenate([x, concat_with], axis=3)

        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name + "_convB")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', name=name + "_convC")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        return x

    def build_model(self):
        i = Input(shape=self.input_shape)
        # encoder
        conv1 = self.downsampling_block(i, 32)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self.downsampling_block(pool1, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self.downsampling_block(pool2, 128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = self.downsampling_block(pool3, 256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # bottleneck
        conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
        conv5 = LeakyReLU(alpha=0.1)(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = LeakyReLU(alpha=0.1)(conv5)

        # decoder
        conv6 = self.upsampling_block(conv5, 256, "up1", concat_with=conv4)
        conv7 = self.upsampling_block(conv6, 128, "up2", concat_with=conv3)
        conv8 = self.upsampling_block(conv7, 64, "up3", concat_with=conv2)
        conv9 = self.upsampling_block(conv8, 32, "up4", concat_with=conv1)

        # output
        o = Conv2D(filters=1, kernel_size=3, strides=(1, 1), activation='sigmoid', padding='same', name='conv10')(conv9)

        model = Model(inputs=i, outputs=o)
        return model

    def depth_loss(self,y_true, y_pred):
        w1, w2, w3 = 1.0,1.0,0.2

        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

    def depth_acc(self,y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def poly_decay(self, epoch):
        maxEpochs = self.EPOCHS
        baseLR = self.INIT_LR
        power = 1.0
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        return alpha

    def grid_search(self, pipeline, param_grid, verbose=True, save_best_model=False, save_path=None):
        """
        Custom grid search implementation using the traditional fit method with generators.

        Args:
        - param_grid: Dictionary specifying the hyperparameter grid.
        - verbose: Whether to print progress messages.
        - save_best_model: Whether to save the best model found during grid search.
        - save_path: Path to save the best model.

        Returns:
        - best_model: The best model found during grid search.
        - best_params: The best hyperparameters found.
        """
        best_model = None
        best_score = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            if verbose:
                print("Training with parameters:", params)

            self.model = self.build_model()
            # Train the model
            pipeline.generator = tf.data.Dataset.from_generator(
                output_signature=pipeline.output_signature,
                generator=pipeline.execute_pipeline_once,
                args=(params['batch_size'],)
            )

            if params['optimizer'] == 'adam':
                optimizer = Adam(learning_rate=params['learning_rate'])
            elif params['optimizer'] == 'rmsprop':
                optimizer = RMSprop(learning_rate=params['learning_rate'])
            else:
                optimizer = SGD(learning_rate=params['learning_rate'])

            callbacks = [LearningRateScheduler(self.poly_decay)]
            self.model.compile(optimizer=optimizer, loss=self.depth_loss, metrics=[self.depth_acc])
            history = self.model.fit(pipeline.generator, epochs=params['epochs'], callbacks=callbacks)

            # Evaluate the model
            score = history.history['loss'][-1]

            if score < best_score:
                best_score = score
                best_params = params
                best_model = self.model

            if verbose:
                print("  - LOSS:", score)

        if verbose:
            print("Best parameters:", best_params)
            print("Best score:", best_score)

        # Save the best model if specified
        if save_best_model and best_model is not None and save_path is not None:
            best_model.save(save_path)
            if verbose:
                print("Best model saved to:", save_path)

        return best_model, best_params

    def predict(self, batch_of_inputs):
        """
        Predict depth maps for a batch of input images.

        Args:
        - batch_of_inputs: numpy array of shape (batch_size, height, width, channels)

        Returns:
        - predictions: numpy array of shape (batch_size, height, width, 1)
        """
        predictions = self.model.predict(batch_of_inputs)
        return predictions

    def evaluate(self, batch_of_inputs, batch_of_labels):
        """
        Evaluate the model on a batch of input images and corresponding labels.

        Args:
        - batch_of_inputs: numpy array of shape (batch_size, height, width, channels)
        - batch_of_labels: numpy array of shape (batch_size, height, width, 1)

        Returns:
        - loss: Scalar value representing the mean squared error loss on the batch.
        """
        loss = self.model.evaluate(batch_of_inputs, batch_of_labels)
        return loss
