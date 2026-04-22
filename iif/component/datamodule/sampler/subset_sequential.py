import torch


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Adapted from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, indices):
        """
        Creates new sampler
        :param indices: The indices to sample from sequentially
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)