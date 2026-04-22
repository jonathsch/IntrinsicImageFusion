from enum import Enum
from dataclasses import dataclass


class TrainStage(Enum):
    """Definition of the different training stages"""
    Training: str = "train"
    Validation: str = "val"
    Test: str = "test"

    def is_train(self):
        """
        Checks whether the stage referes to a training stage or not
        :return: True if the stage is Training or Validation
        """
        return self == self.Training

    def __str__(self):
        return self.value

    def from_str(self, val):
        return TrainStage(val)
