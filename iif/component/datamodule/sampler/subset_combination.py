from itertools import combinations_with_replacement

import torch


class SubsetCombinationalSampler(torch.utils.data.Sampler):
    def __init__(self, indices, num_combinations):
        self.num_combinations = num_combinations
        self.indices = [index for tuple_index in combinations_with_replacement(indices, 2) for index in tuple_index]

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)