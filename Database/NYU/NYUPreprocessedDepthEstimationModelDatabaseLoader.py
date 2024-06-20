import numpy as np

from Database.DatabaseProcess.DatabaseLoader import DatabaseLoader
import os
from PIL import Image


class NYUPreprocessedDepthEstimationModelDatabaseLoader(DatabaseLoader):
    def __init__(self, path, sample_container_paths, verbose=False):
        super().__init__(path, sample_container_paths, verbose=verbose)

        # Register sample_container opener
        self.add_sample_container_opener(0, self.open_preprocessed_folder)

        # Register sample getter
        self.add_sample_getter(0, self.get_random_sample)

    def open_preprocessed_folder(self, folder_path):
        """
        Receives a folder path and returns the folder path and the number of folders inside.
        The folders inside will be named as numbers going from 0 to N. Returns n along with the folder itself.

        Args:
        - folder_path (str): Path to the main folder.

        Returns:
        - tuple: The folder path and the number of subfolders inside.
        """
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        num_subfolders = len(subfolders)
        return folder_path, num_subfolders

    def get_random_sample(self,folder_path, sample):
        """
        Receives a folder path and a sample number, then opens the corresponding folder where
        there will be these files: original.png, gaussian_X (X from 1 to 4).png, border_detection.png,
        label.png, edge_detection.png, gradient_magnitude.png, and gradient_orientation.png.
        Stores them in a dict and returns it.

        Args:
        - folder_path (str): Path to the main folder.
        - sample (int): The sample number to open.

        Returns:
        - dict: A dictionary with the images.
        """
        sample_folder = os.path.join(folder_path, str(sample))
        image_files = {
            'original': 'original.png',
            'label': 'label.png',
        }

        images = {}
        for key, filename in image_files.items():
            image_path = os.path.join(sample_folder, filename)
            if os.path.exists(image_path):
                images[key] = Image.open(image_path)
                images[key] = np.array(images[key])
            else:
                images[key] = None

        return images
