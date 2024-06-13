import os
from Database.DatabaseProcess.DatabaseProcess import DatabaseProcess


class DatabaseSaver(DatabaseProcess):
    def __init__(self, database_path, folder_paths, verbose=True):
        """
        Initialize the DatabaseLoader.

        Args:
        - database_path (str): Path to the directory where the database will be stored.
        - folder_paths (list of str): List of sample_container paths specifying the main folders in the database.
        - verbose (bool): Whether to print verbose messages. Default is True.
        """
        super().__init__(verbose=verbose)
        self.database_path = database_path
        self.folder_paths = folder_paths
        self.example_creator = {}

    def process_batch(self, input_data=None, current_batch_index=-1):
        current_batch_index = 0
        while True:
            if input_data is None:
                yield None
            self.store(input_data, current_batch_index)
            current_batch_index += 1
            yield input_data

    def add_example_creator(self, file_index, example_creator_func):
        """
        Add an example creator function to the DatabaseLoader.

        Args:
        - example_creator_func (function): Function to be called to create examples in the database.
        """
        self.example_creator[file_index] = example_creator_func

    def store(self, data, current_batch_index):
        """
        Store the database by calling the registered example creator functions.

        Args:
        - data (list of list): The database data MUST be [FILE][SAMPLE]
        """
        # Create main folders
        if current_batch_index == 0:
            for folder_path in self.folder_paths:
                folder_full_path = os.path.join(self.database_path, folder_path)
                os.makedirs(folder_full_path, exist_ok=True)
                if self.verbose:
                    print(f"Created folder: {folder_full_path}")

        # Iterate over the data and call the example creator functions
        batch_size = len(data[0])
        for file_index, file in enumerate(data):
            for sample_number, sample_data in enumerate(file):
                # Call the example creator function for the current sample_container and sample
                folder_full_path = os.path.join(self.database_path, self.folder_paths[file_index] + "\\" + str(sample_number + batch_size * current_batch_index))
                os.makedirs(folder_full_path, exist_ok=True)
                self.example_creator[file_index](folder_full_path, sample_data)
                if self.verbose:
                    print(f"Stored example {sample_number + batch_size * current_batch_index} in folder {self.folder_paths[file_index]}")
