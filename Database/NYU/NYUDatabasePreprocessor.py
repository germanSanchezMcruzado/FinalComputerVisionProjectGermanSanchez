from Database.DatabaseProcess.DatabasePreprocessor import DatabasePreprocessor
from Preprocessing.ImagePreprocessor import ImagePreprocessor


class NYUDatabasePreprocessor(DatabasePreprocessor):
    def __init__(self):
        super().__init__()

    def process_batch(self,input_data=None, current_batch_index=-1):
        """
        Process the database
        Args:
        - input_data (list of list): The database data MUST be [FILE][SAMPLE]
        """

        processed_data = []
        for i, file in enumerate(input_data):
            file_data = []
            for j, sample in enumerate(file):
                sample_data = []
                label_image = sample['depths']
                original_image = sample['images']
                sample_data.append(("original", original_image))
                sample_data.append(("label", label_image))

                original_image = ImagePreprocessor.noise_reduction(original_image)

                aux_img = ImagePreprocessor.gradient_orientation(
                    *ImagePreprocessor.partial_derivatives(original_image))
                sample_data.append(("gradient_orientation", aux_img))

                aux_img = ImagePreprocessor.gradient_magnitude(original_image)
                sample_data.append(("gradient_magnitude", aux_img))

                aux_img = ImagePreprocessor.edge_detection(original_image)
                sample_data.append(("edge_detection", aux_img))

                aux_img = ImagePreprocessor.corner_detection(original_image)
                sample_data.append(("corner_detection", aux_img))

                aux_img = ImagePreprocessor.noise_reduction(original_image, ksize=(9, 9))
                sample_data.append(("gaussian_1", aux_img))

                aux_img = ImagePreprocessor.noise_reduction(original_image, ksize=(15, 15))
                sample_data.append(("gaussian_2", aux_img))

                aux_img = ImagePreprocessor.noise_reduction(original_image, ksize=(21, 21))
                sample_data.append(("gaussian_3", aux_img))

                aux_img = ImagePreprocessor.noise_reduction(original_image, ksize=(31, 31))
                sample_data.append(("gaussian_4", aux_img))

                aux_img = ImagePreprocessor.kmeans_image(original_image, num_clusters=8)
                sample_data.append(("kmeans", aux_img))

                file_data.append(sample_data)

            processed_data.append(file_data)

        return processed_data
