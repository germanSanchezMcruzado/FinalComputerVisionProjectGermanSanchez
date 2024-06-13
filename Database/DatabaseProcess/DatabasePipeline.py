import numpy as np
import tensorflow as tf

import Database.DatabaseProcess.DatabaseLoader


class DatabasePipeline():
    def __init__(self, processes, output_signature, batch_size):
        super().__init__()
        self.processes = processes
        self.batch_size = batch_size
        self.output_signature = output_signature
        self.generator = tf.data.Dataset.from_generator(output_signature=output_signature, generator=self.execute_pipeline_once, args=(self.batch_size,))

    def execute_pipeline_once(self, batch_size):
        """
        Process one batch
        Args:
        - input_data (list of list): The database data MUST be [FILE][SAMPLE]
        """
        current_batch_index = 0
        self.processes[0].batch_size = batch_size

        while True:
            output_data = self.processes[0].process_batch(None, current_batch_index)
            for i in range(1, len(self.processes)):

                process = self.processes[i]
                if isinstance(process, Database.DatabaseProcess.DatabaseLoader.DatabaseLoader) and current_batch_index == 0:
                    process.batch_size = np.array(output_data).shape[1]

                output_data = process.process_batch(output_data, current_batch_index)

            if output_data is None:
                break
            current_batch_index += 1
            yield output_data
