import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

# Data Visualization
from DataVisualizer.DataVisualizer import DataVisualizer

# Database Processing and Pipeline
from Database.DatabaseProcess.DatabasePipeline import DatabasePipeline

# NYU Database Related Imports
from Database.NYU.NYUPreprocessedDepthEstimationModelDatabaseLoader import NYUPreprocessedDepthEstimationModelDatabaseLoader
from Database.NYU.NYUPreprocessedDepthEstimationModelDatabasePreprocessor import NYUPreprocessedDepthEstimationModelDatabasePreprocessor

# Model
from Models.DepthEstimationModel import DepthEstimationModel


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def get_nyu_preprocessed_depth_estimation_database_pipeline():
    nyu_preprocessed_depth_estimation_database_loader = NYUPreprocessedDepthEstimationModelDatabaseLoader(
        path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Data\PreprocessedNYU",
        sample_container_paths=["PreprocessedImages"],
        verbose=False
    )

    nyu_preprocessed_depth_estimation_database_preprocessor = NYUPreprocessedDepthEstimationModelDatabasePreprocessor(verbose=False)

    nyu_preprocessed_database_pipeline = DatabasePipeline(
        [nyu_preprocessed_depth_estimation_database_loader, nyu_preprocessed_depth_estimation_database_preprocessor],
        output_signature=(
            tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32, name=None),
            tf.TensorSpec(shape=(None, 480, 640,1), dtype=tf.float32, name=None)
        )
    )

    return nyu_preprocessed_database_pipeline


def create_depth_estimation_model():
    # Define the model input shapes based on the dataset
    input_shape_original = (480, 640, 3)

    model = DepthEstimationModel(input_shape=input_shape_original)

    database_pipeline = get_nyu_preprocessed_depth_estimation_database_pipeline()
    final_model, final_params = model.grid_search(database_pipeline, param_grid={
            'optimizer': ['adam'],  # Different optimizers to try
            'batch_size': [3],
            'learning_rate': [0.01],  # Different learning rates
            'epochs': [5]  # Different numbers of epochs
        }, save_best_model=True, save_path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Models\SavedModels\DepthEstimation")
    return


def test_depth_estimation_model():
    model = DepthEstimationModel(saved_path=r"C:\Users\germa\PycharmProjects\ComputerVisionProject\Models\SavedModels\DepthEstimation")
    database_pipeline = get_nyu_preprocessed_depth_estimation_database_pipeline()
    generator = iter(database_pipeline.generator)
    for i in range(10):
        data = next(generator)
        predictions = model.predict(data[0])

        for j in range(len(data)):
            pred = predictions[j]
            pred = np.squeeze(pred,axis=-1)

            gt = data[1][j]

            # Plotting
            plt.subplot(1, 2, 1)
            plt.title('Prediction')
            plt.axis("off")
            plt.imshow(pred, cmap=plt.get_cmap('inferno_r'))

            plt.subplot(1, 2, 2)
            plt.title('Ground Truth')
            plt.axis("off")
            gt = np.squeeze(gt, axis=-1)
            plt.imshow(gt, cmap=plt.get_cmap('inferno_r'))

            plt.show()




if __name__ == "__main__":
    create_depth_estimation_model()
    test_depth_estimation_model()

