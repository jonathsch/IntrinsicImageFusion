from torch.utils.data import Dataset

from iif.utils.datastructure import Batch
from iif.utils.stage import TrainStage


class ComposedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        sample = Batch()

        for _, dataset in self.datasets.items():
            sample.update(dataset[index])

        return sample

    def __len__(self):
        """
        Returns the maximum length of all datasets
        :return:
        """
        return max((len(dataset) for dataset in self.datasets.values()))
