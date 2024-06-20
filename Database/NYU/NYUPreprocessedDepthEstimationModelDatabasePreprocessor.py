import random
import cv2
import imutils
import tensorflow as tf
from matplotlib import pyplot as plt

from Database.DatabaseProcess.DatabasePreprocessor import DatabasePreprocessor

import numpy as np
import cv2

class NYUPreprocessedDepthEstimationModelDatabasePreprocessor(DatabasePreprocessor):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def preprocess_depth_map(self, depth_map, horizontal_flip=False):
        depth_map = self.normalize_img(depth_map.astype("float"))
        depth_map = self.normalize_img(depth_map)

        if horizontal_flip:
            depth_map = cv2.flip(depth_map, 1)

        depth_map = np.reshape(depth_map, (depth_map.shape[0], depth_map.shape[1], 1))
        return depth_map

    def process_batch(self, input_data=None, current_batch_index=-1):
        """
        Process the database
        Args:
        - input_data (list of list): The database data MUST be [FILE][SAMPLE]
        """
        if input_data is None:
            return None

        original_images_batch = []
        label_images_batch = []

        for i, file in enumerate(input_data):
            for j, sample in enumerate(file):
                if self.verbose:
                    print(f"Processing sample_container: {i} Sample: {j}")

                # Normalize and process the original and label images
                original_image = cv2.divide(sample['original'].astype("float"),255.0)
                if random.choice([True, False]):
                    original_image = cv2.flip(original_image, 1)

                label_image = self.preprocess_depth_map(sample['label'], random.choice([True, False]))

                original_images_batch.append(original_image)

                label_images_batch.append(label_image)

        return (np.array(original_images_batch), np.array(label_images_batch))

    def normalize_img(self,img):
        norm_img = (img-img.min()) / (img.max()-img.min() + 1e-3)
        return norm_img

