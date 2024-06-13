import os

from Database.DatabaseProcess.DatabaseProcess import DatabaseProcess


class DatabaseLoader(DatabaseProcess):
    def __init__(self, path, sample_container_paths, batch_size=10, verbose=True):
        super().__init__(verbose=verbose)
        self.path = path
        self.sample_containers_paths = sample_container_paths
        self.batch_size = batch_size
        self.sample_container_sizes = []
        self.sample_container_openers = {}  #how to open the folder or sample_container containing samples
        self.sample_getters = {}  #how to store the samples opened by sample openers

    def process_batch(self, input_data=None, current_batch_index=-1):
        sample_indices = [(current_batch_index * self.batch_size) + i for i in range(self.batch_size)]
        data = []

        for i, container in enumerate(self.sample_containers_paths):
            container_samples = []
            sample_container, sample_container_size = self.open_sample_container(i)

            if current_batch_index + 1 >= self.sample_container_sizes[i] // self.batch_size:
                return None

            for sample_index in sample_indices:
                if sample_index < sample_container_size:
                    container_samples.append(self.get_sample(sample_container, i, sample_index))
                else:
                    wrapped_index = sample_index % sample_container_size
                    container_samples.append(self.get_sample(sample_container, i, wrapped_index))

            data.append(container_samples)

        current_batch_index += 1
        return data

    def add_sample_container_opener(self, sample_container_index, opener_function):
        """
        Add a function pointer to open the sample_container specified by the index.
        """
        self.sample_container_openers[sample_container_index] = opener_function
        self.sample_container_sizes = []

        for i, sample_container_path in enumerate(self.sample_containers_paths):
            sample_container, sample_container_size = self.open_sample_container(i)
            self.sample_container_sizes.append(sample_container_size)

    def add_sample_getter(self, sample_container_index, getter_function):
        """
        Add a function pointer to get a sample from the sample_container specified by the index.
        """
        self.sample_getters[sample_container_index] = getter_function

    def open_sample_container(self, sample_container_index):
        """
        Open the sample_container specified by the index using the corresponding opener function.
        """
        if sample_container_index < 0 or sample_container_index >= len(self.sample_containers_paths):
            if self.verbose:
                print("Error: Invalid sample_container index.")
            return None
        else:
            opener_function = self.sample_container_openers.get(sample_container_index)
            if not opener_function:
                if self.verbose:
                    print("Error: No opener function registered for sample_container {}.".format(sample_container_index))
                return None
            else:
                file_path = os.path.join(self.path, self.sample_containers_paths[sample_container_index])
                if self.verbose:
                    print("Opening sample_container:", file_path)
                return opener_function(file_path)

    def get_sample(self, sample_container, sample_container_index, sample):
        """
        Get a sample from the sample_container or folder specified by the index using the corresponding getter function.
        input: sample_container --> sample_container to get sample from
               sample_container_index --> index of the sample_container to get sample from
               sample --> sample number
        """
        if sample_container_index < 0 or sample_container_index >= len(self.sample_containers_paths):
            if self.verbose:
                print("Error: Invalid sample_container index.")
            return None
        else:
            getter_function = self.sample_getters.get(sample_container_index)
            if not getter_function:
                if self.verbose:
                    print("Error: No getter function registered for sample_container {}.".format(sample_container_index))
                return None
            else:
                if self.verbose:
                    print("Getting sample {} from sample_container {}.".format(sample, sample_container_index))
                return getter_function(sample_container, sample)
