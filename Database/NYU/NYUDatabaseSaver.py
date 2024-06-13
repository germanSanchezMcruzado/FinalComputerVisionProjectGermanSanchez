from Database.DatabaseProcess.DatabaseSaver import DatabaseSaver
import cv2
import os


class NYUDatabaseSaver(DatabaseSaver):
    def __init__(self, database_path, folder_paths, verbose=True):
        """
        Initialize the NYUDatabaseSaver.

        Args:
        - database_path (str): Path to the directory where the database will be stored.
        - folder_path (str): Path specifying the main folder in the database.
        - verbose (bool): Whether to print verbose messages. Default is True.
        """
        super().__init__(database_path, folder_paths, verbose=verbose)

        super().add_example_creator(0, example_creator_func=self.example_creator_func)

    def example_creator_func(self, folder_path, sample_data):
        """
        Create and store examples in the database.

        Args:
        - folder_path (str): Path to the folder where the examples will be stored.
        - sample_data (list of tuple): List of tuples containing image data and their names.
        """
        # Iterate over the sample data (tuples of image and name)
        for name, image in sample_data:
            # Save the image to the folder with the given name
            image_path = os.path.join(folder_path, name + ".png")
            cv2.imwrite(image_path, image)
            if self.verbose:
                print(f"Stored image '{name}' in folder {folder_path}")
