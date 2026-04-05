from functools import partial
from typing import Optional, Any

import hydra
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate

from iif.utils.config import range2list
from iif.utils.dataloading import collate_fn_map
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger
from iif.utils.stage import TrainStage


class DataModule(LightningDataModule):
    DEFAULT_SAMPLING_CFG = {
        "num_workers": 4,
        "batch_size": 4,
        "shuffle": True,
    }

    def __init__(self,
                 dataset_cfg: str,
                 sampling_cfg: Optional[dict] = dict(),
                 stage: TrainStage = TrainStage.Training,):
        super().__init__()
        self.logger = init_logger()

        self.dataset_cfg = dataset_cfg
        self.sampling_cfg = self.DEFAULT_SAMPLING_CFG
        self.sampling_cfg.update(sampling_cfg)

        self.stage = stage if isinstance(stage, TrainStage) else TrainStage(stage)

        self._dataset_train = None
        self._dataset_valid = None
        self._dataset_test = None

        self.sampler_type = None

        self.collate_fn = partial(collate, collate_fn_map=collate_fn_map)

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            self.dataset_train = self._load_dataset(mode=TrainStage.Training)
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, dataset_train):
        self._dataset_train = dataset_train
        self.logger.info(f"Training dataset set to {dataset_train}, number of samples: {len(dataset_train)}")

    @property
    def dataset_valid(self):
        if self._dataset_valid is None:
            self.dataset_valid = self._load_dataset(mode=TrainStage.Validation)
        return self._dataset_valid

    @dataset_valid.setter
    def dataset_valid(self, dataset_valid):
        self._dataset_valid = dataset_valid
        self.logger.info(f"Validation dataset set to {dataset_valid}, number of samples: {len(dataset_valid)}")

    @property
    def dataset_test(self):
        if self._dataset_test is None:
            self.dataset_test = self._load_dataset(mode=TrainStage.Test)
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, dataset_test):
        self._dataset_test = dataset_test
        self.logger.info(f"Test dataset set to {dataset_test}, number of samples: {len(dataset_test)}")

    def _load_dataset(self, mode: TrainStage, **kwargs):
        self.logger.debug(f"Loading {mode} dataset! - training: {mode}")
        dataset = self.load_dataset(mode, **kwargs)
        self.logger.debug(f"The {mode} dataset loaded!")
        return dataset

    def load_dataset(self, stage: TrainStage, **kwargs):
        cfg = Batch(self.dataset_cfg)
        stage_keys = list(cfg.query_wildcard("*stage").keys())
        if len(stage_keys) == 0:
            cfg["stage"] = stage
        if len(stage_keys) == 1 and cfg[stage_keys[0]] is None:
            cfg[stage_keys[0]] = stage
        # assert len(stage_keys) > 0, f"Dataset config {self.dataset_cfg} does not contain stage information!"
        # for stage_key in stage_keys:
        #     cfg[stage_key] = stage

        dataset = hydra.utils.instantiate(cfg.to_dict(),
                                          **kwargs)
        return dataset

    def get_dataset(self, stage):
        train_datasets = {
            TrainStage.Training.value: self.dataset_train,
            TrainStage.Validation.value: self.dataset_valid,
            TrainStage.Test.value: self.dataset_test
        }
        if stage in train_datasets:
            return train_datasets[stage]
        else:
            raise ValueError(f"Unknown stage {self.stage}")

    def prepare_data(self):
        # Called only on 1 GPU
        self.logger.debug(f"Preparing data!")
        if self.dataset_train is None:
            raise RuntimeError("Training dataset is not set!")
        self.logger.debug(f"Data prepared!")

    def setup(self, stage: Optional[str] = None):
        # Called on every GPUs
        self.logger.debug(f"Setup data!")
        # Data already split into train/val/test
        self.logger.debug(f"Data set up!")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.stage == TrainStage.Training:
            dataset = self.dataset_train
        else:
            dataset = self.dataset_test
        self.logger.debug(f"Creating train dataloader!")

        sampler = self.sampling_cfg.get("sampler", None)
        if sampler is not None:
            sampler = hydra.utils.instantiate(sampler,
                                              indices=range2list(sampler.pop("indices"),
                                                                 max_length=len(dataset)))
        loader = DataLoader(
            dataset,
            batch_size=self.sampling_cfg["batch_size"],
            num_workers=self.sampling_cfg["num_workers"],
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
            shuffle=self.sampling_cfg["shuffle"] if sampler is None else False,
            collate_fn=self.collate_fn
        )
        self.logger.debug(
            f"Train dataloader created with dataset length: {len(loader.dataset)}, batch sampler length: {len(loader.batch_sampler) if loader.batch_sampler is not None else None}!")
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.stage == TrainStage.Training:
            dataset = self.dataset_valid
        else:
            dataset = self.dataset_test
        self.logger.debug(f"Creating validation dataloader!")
        loader = DataLoader(
            dataset,
            batch_size=self.sampling_cfg["batch_size"],
            num_workers=self.sampling_cfg["num_workers"],
            drop_last=False,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        self.logger.debug(
            f"Validation dataloader created with dataset length: {len(loader.dataset)}, batch sampler length: {len(loader.batch_sampler) if loader.batch_sampler is not None else None}!")
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = self.dataset_test
        self.logger.debug(f"Creating test dataloader!")
        loader = DataLoader(
            dataset,
            batch_size=self.sampling_cfg["batch_size"],
            num_workers=self.sampling_cfg["num_workers"],
            drop_last=False,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        self.logger.debug(
            f"Test dataloader created with dataset length: {len(loader.dataset)}, batch sampler length: {len(loader.batch_sampler) if loader.batch_sampler is not None else None}!")
        return loader

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if hasattr(self.dataset_train, "on_after_batch_transfer"):
            return self.dataset_train.on_after_batch_transfer(batch, dataloader_idx)
        else:
            return batch