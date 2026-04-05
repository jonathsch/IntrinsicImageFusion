import torch
from torch.utils.data import RandomSampler

from iif.utils.logging import init_logger


class SubsetRepeatedCacheSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Adapted from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, indices=None, shuffle=True, seed: int = 0, chunk_size=1, num_chunks_in_cache=1, sub_iterations=1, **kwargs):
        """
        Creates new sampler
        :param indices: The indices to sample from sequentially
        """
        self.chunk_size = chunk_size
        self.seed = seed
        self.shuffle = shuffle
        self.num_chunks_in_cache = max(num_chunks_in_cache, 1)
        self.sub_iterations = sub_iterations

        self.indices = indices

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.module_logger = init_logger()
        self.module_logger.info(f"Sampler chunk size: {self.chunk_size}")

    def _upsample_list(self, indices_list, num_cached_samples):
        upsampled_indices = []
        num_indices = len(indices_list)
        for indices_idx in range(0, num_indices, num_cached_samples):
            upsampled_indices.extend(indices_list[indices_idx:indices_idx + num_cached_samples] * self.sub_iterations)
        return upsampled_indices

    def _shuffle_list(self, list_to_shuffle, chunk_size=1):
        indices_idx = torch.randperm(len(list_to_shuffle) // chunk_size, generator=self.generator).tolist()
        global_indices = []
        for idx in indices_idx:
            global_indices.extend(list_to_shuffle[idx * chunk_size: (idx + 1) * chunk_size])
        if len(global_indices) < len(list_to_shuffle):
            global_indices.extend(list_to_shuffle[len(global_indices):])
        return global_indices

    def _shuffle_upsampled_list(self, indices_list, num_cached_samples):
        shuffled_indices = []
        num_indices = len(indices_list)
        for chunk_start_idx in range(0, num_indices, num_cached_samples):
            shuffled_indices.extend(self._shuffle_list(indices_list[chunk_start_idx:chunk_start_idx + num_cached_samples]))
        return shuffled_indices

    def __iter__(self):
        if self.shuffle:
            # 1. Shuffle the chunks
            global_indices = self._shuffle_list(self.indices, chunk_size=self.chunk_size)

            # 2. Upsample
            global_indices = self._upsample_list(global_indices, num_cached_samples=self.chunk_size * self.num_chunks_in_cache)

            # 3. Shuffle inside the cached_chunks
            global_indices = self._shuffle_upsampled_list(global_indices, num_cached_samples=self.chunk_size * self.num_chunks_in_cache)
        else:
            # 1. Use the indices without shuffling
            global_indices = self.indices

            # 2. Upsample
            global_indices = self._upsample_list(global_indices, num_cached_samples=self.chunk_size * self.num_chunks_in_cache)

        # self.module_logger.info(f"Sampler indices: {global_indices}")

        return iter(global_indices)

    def __len__(self):
        return len(self.indices) * self.sub_iterations
