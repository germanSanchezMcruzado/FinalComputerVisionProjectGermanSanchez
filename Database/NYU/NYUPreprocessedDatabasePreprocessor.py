import numpy as np
import tensorflow as tf

from Database.DatabaseProcess.DatabasePreprocessor import DatabasePreprocessor
from Preprocessing.ImagePreprocessor import ImagePreprocessor


class NYUPreprocessedDatabasePreprocessor(DatabasePreprocessor):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def process_batch(self, input_data=None, current_batch_index=-1):
        """
        Process the database
        Args:
        - input_data (list of list): The database data MUST be [FILE][SAMPLE]
        """
        if input_data is None:
            return None

        for i, file in enumerate(input_data):
            original_images_batch = []
            gaussian_images_batch = []
            label_images_batch = []
            kmeans_images_batch = []
            gradient_orientation_images_batch = []
            gradient_magnitude_images_batch = []
            edge_module_images_batch = []
            for j, sample in enumerate(file):
                if self.verbose:
                    print("Processing sample_container: {i} Sample: {j} ".format(i=i, j=j))

                original_images_batch.append(ImagePreprocessor.normalize_image(sample['original']))
                gaussian_images_batch.append(np.stack([ImagePreprocessor.normalize_image(sample[f'gaussian_{i + 1}']) for i in range(4)], axis=0))
                label_images_batch.append(sample['label'])
                kmeans_images_batch.append(ImagePreprocessor.normalize_image(sample['kmeans']))

                edge_image = np.stack([sample['edge_detection'] for _ in range(3)], axis=-1)
                corner_detection_image = ImagePreprocessor.normalize_image(sample['corner_detection'] + edge_image)
                gradient_magnitude_images_batch.append(
                    ImagePreprocessor.normalize_image(np.stack([sample['gradient_magnitude'] for _ in range(3)], axis=-1)))
                gradient_orientation_images_batch.append(
                    ImagePreprocessor.normalize_image(np.stack([sample['gradient_orientation'] for _ in range(3)], axis=-1)))
                edge_module_images_batch.append(np.stack(
                    [corner_detection_image, kmeans_images_batch[-1], gradient_magnitude_images_batch[-1],
                     gradient_orientation_images_batch[-1], gradient_magnitude_images_batch[-1]], axis=0
                ))

            return (((gaussian_images_batch, edge_module_images_batch, original_images_batch)
                     , label_images_batch))
