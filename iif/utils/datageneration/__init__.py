import os

from iif.utils.datageneration.distributed import DistributedDataGenerator, DataGenerationState


class ExampleGenerator(DistributedDataGenerator):
    def _prepare_dataset(self, storage, dist_meta):
        """
        This method is called when the dataset is prepared for the first time.
        :param storage: The storage instance to save the metadata to.
        :param dist_meta: The distributed metadata to be saved.
        """
        # Prepare the metadata storing the required information for processing a sample
        metadata = {
            "sample_param": [],
        }

        # Collect the sample params
        for idx in range(10):
            metadata["sample_param"].append(idx)
            dist_meta["states"].append(DataGenerationState.NOT_GENERATED.value)

        return metadata, dist_meta

    def collect_sample_metadata(self, storage, sample_id_to_render):
        """
        This method is called to collect the metadata required to generate a single sample.
        :param storage: The storage instance where the metadata is saved.
        :param sample_id_to_render: The id of the sample to be rendered.
        :return: The metadata required to generate the sample.
        """
        sample_param = storage["/sample_param"][sample_id_to_render]
        return {"sample_param": sample_param}

    def generate_sample(self, sample_metadata):
        """
        This method is called to generate a single sample.
        :param sample_metadata: The metadata required to generate the sample.
        :return: The sample itself.
        """
        self.module_logger.info(f"Generate sample for {sample_metadata}!")

        out_folder_path = os.path.join(self.output_dir, self.SUB_FOLDER)
        os.makedirs(out_folder_path, exist_ok=True)

        # TODO: Generate the sample
        self.module_logger.info(f"Sample {sample_metadata} generated!")

        sample_info = {"sample": None}

        return sample_info

    def save_sample(self, storage, sample, sample_metadata):
        """
        Used if writing to the storage, which is used to synchronize
        :param storage:
        :param sample:
        :param sample_metadata:
        :return:
        """
        pass


if __name__ == "__main__":
    generator = ExampleGenerator(output_dir="data/example")
    generator.prepare_dataset()
    generator.generate_samples()
