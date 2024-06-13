import mat73
import tensorflow as tf

from Database.DatabaseProcess.DatabaseLoader import DatabaseLoader


class NYUDatabaseLoader(DatabaseLoader):
    def __init__(self, path, sample_container_paths, verbose=False):
        super().__init__(path, sample_container_paths, verbose=verbose)

        # Register sample_container opener
        self.add_sample_container_opener(0, self.open_mat_file)

        # Register sample getter
        self.add_sample_getter(0, self.get_random_sample)

    def open_mat_file(self, file_path):
        """
        Open the .mat sample_container and return its contents, sample_container size, and number of samples for each key.
        """
        try:
            data = mat73.loadmat(file_path)
            return data, len(data)  # Returning the data and the number of samples (length of keys)
        except Exception as e:
            if self.verbose:
                print("Error: Failed to open .mat sample_container:", e)
            return None, None

    def get_random_sample(self, file, sample):
        """
        Get the Nth sample (row) from each data type in the .mat sample_container.
        """
        if file is not None:
            random_sample = {"depths": file["depths"][:, :, sample], "images": file["images"][:, :, :, sample]}
            return random_sample
        else:
            return None
