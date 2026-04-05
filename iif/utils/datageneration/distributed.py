import shutil
from abc import abstractmethod, ABC
import os
from enum import Enum
import uuid

import numpy as np
from omegaconf import OmegaConf

from iif.utils.logging import init_logger, log_to_file
from iif.utils.storage.hdf5 import HDF5StorageBackend


class DataGenerationState(Enum):
    NOT_GENERATED = 0
    IN_PROGRESS = 1
    GENERATED = 2


class DistributedDataGenerator(ABC):
    """
    Abstract class handling the dataset generation and preparation.
    This class supports distributed generation of the dataset.
    Any number of processes can be started all saving to the same folder.
    These workers are automatically synchronized by using a shared metadata file as semaphore.
    """
    STATUS_FILE = "status.hdf5"
    CONFIG_FILE = "config.yaml"
    SUB_FOLDER = ""
    LOG_FILE = f"log_{os.getpid()}_{str(uuid.uuid4())}.txt"

    def __init__(self,
                 output_dir="",
                 batch_size=1,
                 finish=False,
                 retry=False,
                 overwrite=False):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.finish = finish
        self.retry = retry
        self.overwrite = overwrite

        self.module_logger = init_logger()
        log_to_file(os.path.join(self.output_dir, self.SUB_FOLDER, self.LOG_FILE))

        self.storage = HDF5StorageBackend(os.path.join(self.output_dir, self.SUB_FOLDER, self.STATUS_FILE), required_fields=["/states"])
        self._num_samples = None

    @property
    def config_file(self):
        return os.path.join(self.output_dir, self.SUB_FOLDER, self.CONFIG_FILE)

    def log_config(self, cfg):
        with open(self.config_file, "w") as f:
            OmegaConf.save(config=cfg, f=f)

    @abstractmethod
    def _prepare_dataset(self, storage, dist_meta):
        """
        This method is called when the dataset is prepared for the first time.
        :param storage: The storage instance to save the metadata to.
        :param dist_meta: The distributed metadata to be saved.
        """
        pass

    def log_preparation_statistics(self, logger=None):
        """
        Callback method to log the preparation statistics.
        :param logger: Logger to use for logging.
        """
        pass

    @abstractmethod
    def collect_sample_metadata(self, storage, sample_id_to_render):
        """
        This method is called to collect the metadata required to generate a single sample.
        :param storage: The storage instance where the metadata is saved.
        :param sample_id_to_render: The id of the sample to be rendered.
        :return: The metadata required to generate the sample.
        """
        pass

    @abstractmethod
    def generate_sample(self, sample_metadata):
        """
        This method is called to generate a single sample.
        :param sample_metadata: The metadata required to generate the sample.
        :return: The sample itself.
        """
        pass

    @abstractmethod
    def save_sample(self, storage, sample, sample_metadata):
        """
        This method is called to save a single sample.
        :param storage: The storage instance where the sample might be saved.
        :param sample: The generated sample.
        :param sample_metadata: The metadata required to generate the sample.
        """
        pass

    def log_generated_samples(self, sample, sample_id_to_render, logger=None):
        """
        Callback method to log the generated sample, called after every sample.
        :param sample: The generated sample.
        :param sample_id_to_render: The id of the sample that has been rendered.
        :param logger: Logger to use for logging.
        """
        pass

    def mark_for_retry(self, storage):
        pass

    def post_process_dataset(self, storage):
        pass

    def prepare_dataset(self, logger=None):
        self.module_logger.info(f"Preparing dataset at {self.output_dir}!")
        if self.finish:
            self.module_logger.info("Finishing in-progress samples!")

        if self.overwrite and self.storage.exists():
            self.module_logger.info("Overwriting the dataset!")
            shutil.rmtree(os.path.join(self.output_dir, self.SUB_FOLDER))

        if not self.storage.exists():
            try:
                with self.storage as storage:
                    self.module_logger.info(f"Dataset at {self.output_dir} not found yet, creating now!")
                    dist_meta = {
                        "states": [],
                        "last_rendered_sample": -1,
                    }
                    metadata, dist_meta = self._prepare_dataset(storage, dist_meta)
                    self.module_logger.info(f"Saving prepared metadata!")
                    self._save_prepared_metadata(storage, metadata)
                    self._save_dist_metadata(storage, dist_meta)
                    self.module_logger.info("Prepared metadata saved!")

            except FileExistsError:
                self.module_logger.info(f"Dataset at {self.output_dir} has been created in meanwhile!")
            self.module_logger.info(f"Dataset at {self.output_dir} created!")
        else:
            self.module_logger.info(f"Dataset at {self.output_dir} already prepared!")

        self.log_preparation_statistics(logger)

    def _save_prepared_metadata(self, storage, metadata):
        for key, value in metadata.items():
            if isinstance(value, list):
                storage.create_dataset(key, data=value)
                self.module_logger.info(f"Dataset {key} has been created with {len(value)} elements!")
            elif isinstance(value, np.ndarray):
                storage.create_dataset(key, data=value)
                self.module_logger.info(f"Dataset {key} has been created with {len(value)} elements!")
            else:
                raise NotImplementedError("Only lists are supported for now!")

    def _save_dist_metadata(self, storage, dist_meta):
        storage.create_dataset("last_rendered_sample", data=dist_meta["last_rendered_sample"])
        storage.create_dataset("states", data=dist_meta["states"])

    def generate_samples(self, logger=None):
        if self.retry:
            with self.storage as storage:
                self.module_logger.info("Retrying samples left in progress state!")
                states = storage["states"][()]
                states[states == DataGenerationState.IN_PROGRESS.value] = DataGenerationState.NOT_GENERATED.value
                storage["states"][()] = states
                storage["last_rendered_sample"][()] = -1

                self.module_logger.info("Retrying custom samples!")
                self.mark_for_retry(storage)
        else:
            while True:
                # Find the next sample to be rendered
                with self.storage as storage:
                    try:
                        sample_ids_to_render = self.find_next_sample_to_generate(storage)
                        sample_metadatas = []
                        for sample_id_to_render in sample_ids_to_render:
                            storage["states"][sample_id_to_render] = DataGenerationState.IN_PROGRESS.value
                            sample_metadatas.append(self.collect_sample_metadata(storage, sample_id_to_render))
                        storage["last_rendered_sample"][()] = sample_ids_to_render[-1]
                        self.module_logger.info(f"Sample found to be generated - {sample_ids_to_render}/{self._num_samples - 1}!")
                    except StopIteration:
                        self.module_logger.info(f"No more samples found to be rendered!")
                        break

                # Render
                for sample_id_to_render, sample_metadata in zip(sample_ids_to_render, sample_metadatas):
                    sample = self.generate_sample(sample_metadata)

                # Save
                with self.storage as storage:
                    for sample_id_to_render in sample_ids_to_render:
                        storage["states"][sample_id_to_render] = DataGenerationState.GENERATED.value

                    # # Save
                    # with self.storage as storage:
                    #     self.save_sample(storage, sample, sample_metadata)
                    #     storage["states"][sample_id_to_render] = DataGenerationState.GENERATED.value
                    #     self.module_logger.info(f"Sample generated - {sample_id_to_render}/{self._num_samples - 1}!")

                    self.log_generated_samples(sample, sample_id_to_render, logger)
            # Post-processing
            with self.storage as storage:
                self.module_logger.info("Post-processing the dataset!")
                self.post_process_dataset(storage)
                self.module_logger.info("Dataset post-processed!")

    def find_next_sample_to_generate(self, storage):
        # Find the first not rendered sample
        states = storage["states"]
        last_rendered_sample = storage["last_rendered_sample"][()]

        if self._num_samples is None:
            self._num_samples = len(states)

        if not self.finish:
            # sample_to_render = list(states).index(RenderingState.NOT_RENDERED.value)
            searched_state = DataGenerationState.NOT_GENERATED
        else:
            searched_state = DataGenerationState.IN_PROGRESS
            last_rendered_sample = -1

        samples_to_generate = []
        for _ in range(self.batch_size):
            try:
                sample_to_generate = next(
                    (idx + last_rendered_sample + 1 for idx, state in enumerate(states[last_rendered_sample + 1:]) if
                    state == searched_state.value))
            except StopIteration:
                break
            samples_to_generate.append(sample_to_generate)
            last_rendered_sample = sample_to_generate

        if len(samples_to_generate) == 0:
            raise StopIteration("No more samples to generate!")

        return samples_to_generate
