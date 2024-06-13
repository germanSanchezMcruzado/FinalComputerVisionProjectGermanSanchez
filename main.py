import tensorflow as tf
import numpy as np
import os

from DataVisualizer.DataVisualizer import DataVisualizer
from Database.DatabaseProcess.DatabasePipeline import DatabasePipeline
from Database.NYU.NYUDatabaseLoader import NYUDatabaseLoader
from Database.NYU.NYUDatabaseSaver import NYUDatabaseSaver
from Database.NYU.NYUDatabasePreprocessor import NYUDatabasePreprocessor
from Database.NYU.NYUPreprocessedDatabasePreprocessor import NYUPreprocessedDatabasePreprocessor
from Database.NYU.NYUPreprocessedDatabaseLoader import NYUPreprocessedDatabaseLoader
from Models.DepthEstimationModel import DepthEstimationModel
from Preprocessing.ImagePreprocessor import ImagePreprocessor

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def get_nyu_preprocessed_database_pipeline():
    nyu_preprocessed_database_loader = NYUPreprocessedDatabaseLoader(
        path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Data\PreprocessedNYU",
        sample_container_paths=["PreprocessedImages"],
        verbose=False
    )

    nyu_preprocessed_database_preprocessor = NYUPreprocessedDatabasePreprocessor(verbose=False)

    nyu_preprocessed_database_pipeline = DatabasePipeline(
        [nyu_preprocessed_database_loader, nyu_preprocessed_database_preprocessor],
        output_signature=(
            ((tf.TensorSpec(shape=(None, 4, 480, 640, 3), dtype=tf.float32, name=None),
             tf.TensorSpec(shape=(None, 5, 480, 640, 3), dtype=tf.float32, name=None),
             tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32, name=None))
             , tf.TensorSpec(shape=(None, 480, 640), dtype=tf.float32, name=None))
        ), batch_size=10
    )

    return nyu_preprocessed_database_pipeline

def get_nyu_database_pipeline():
    nyu_database_loader = NYUDatabaseLoader(
        path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Data\NYU",
        sample_container_paths=["nyu_depth_v2_labeled.mat"]
    )

    nyu_database_saver = NYUDatabaseSaver(
        database_path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Data\PreprocessedNYU",
        folder_paths=["PreprocessedImages"]
    )

    nyu_database_preprocessor = NYUDatabasePreprocessor()

    #nyu_database_pipeline = DatabasePipeline([nyu_database_loader, nyu_database_preprocessor, nyu_database_saver])

    #return nyu_database_pipeline

def create_depth_estimation_model():
    # Define the model input shapes based on the dataset
    input_shape_gaussian = (4, 480, 640, 3)
    input_shape_edge = (5, 480, 640, 3)
    input_shape_original = (480, 640, 3)

    model = DepthEstimationModel(input_shape_gaussian, input_shape_edge, input_shape_original)

    database_pipeline = get_nyu_preprocessed_database_pipeline()
    final_model, final_params = model.grid_search(database_pipeline, param_grid={
            'optimizer': ['adam', 'rmsprop', 'sgd'],  # Different optimizers to try
            'loss': ['mean_squared_error'],
            'batch_size': [8],
            'learning_rate': [0.001, 0.01],  # Different learning rates
            'epochs': [3]  # Different numbers of epochs
        }, save_best_model=True, save_path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Models\SavedModels\DepthEstimation")
    return

def load_and_test_depth_estimation_model(path,pipeline):
    # Define the model input shapes based on the dataset
    input_shape_gaussian = (4, 480, 640, 3)
    input_shape_edge = (5, 480, 640, 3)
    input_shape_original = (480, 640, 3)

    model = DepthEstimationModel(input_shape_gaussian, input_shape_edge, input_shape_original,path)

    test_data = pipeline.execute_pipeline_once(5)
    for batch in test_data:
        inputs, labels = batch

        # Convert inputs to tensors (assuming inputs is a tuple of tensors)
        inputs_tensor1 = tf.convert_to_tensor(inputs[0], dtype=tf.float32)
        inputs_tensor2 = tf.convert_to_tensor(inputs[1], dtype=tf.float32)
        inputs_tensor3 = tf.convert_to_tensor(inputs[2], dtype=tf.float32)

        # Convert labels to tensor
        labels_array = np.array(labels)  # Convert PIL images to numpy arrays
        labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.float32)

        # Evaluate the model
        #loss = model.evaluate((inputs_tensor1, inputs_tensor2, inputs_tensor3), labels_tensor)
        #print(f"Batch Loss: {loss}")

        #Predict
        predictions = model.predict((inputs_tensor1, inputs_tensor2, inputs_tensor3))
        for i,prediction in enumerate(predictions):
            final = tf.squeeze(tf.convert_to_tensor(prediction, dtype=tf.float32),axis = -1)
            DataVisualizer.show_image(final)
            DataVisualizer.show_image(labels_tensor[i])



if __name__ == "__main__":
    create_depth_estimation_model()
    #load_and_test_depth_estimation_model("Models/SavedModels/DepthEstimation", get_nyu_preprocessed_database_pipeline())


