from Database.DatabaseProcess.DatabaseProcess import DatabaseProcess


class DatabasePreprocessor(DatabaseProcess):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def preprocess_batch(self, current_batch_index, input_data=None):
            return None

    def process_batch(self, input_data=None, current_batch_index=-1):
        """
        Process the batch
        Args:
        - input_data (list of list): The database data MUST be [FILE][SAMPLE]
        """
        return self.preprocess_batch(current_batch_index, input_data=input_data)
