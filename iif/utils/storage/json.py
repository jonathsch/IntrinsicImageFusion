import json

import numpy as np

from . import StorageBackend


class JSONStorageBackend(StorageBackend):
    """
    Class handling the access to the dataset. Needs to be thread-safe
    """

    def __init__(self, path, raw=False, readonly=False, timeout=1000):
        super().__init__(path, readonly, timeout)
        self.raw = raw

    def open(self):
        if self.exists():
            storage = json.load(open(self.path, "r"))
            # Decode
            storage = self._numpize_storage(storage)
        else:
            storage = {}
        return storage

    def close(self, storage):
        if not self.readonly:
            storage = self._encode_storage(storage)
            json.dump(storage, open(self.path, "w"), indent=4)

    def __getitem__(self, key):
        # Remove trailing /
        if key.startswith("/"):
            key = key[1:]

        value = self.storage[key]
        return value
        # value = self._str_to_bytes(value)
        # if isinstance(value, (float, int)):
        #     return np.array(self.storage[key])
        # elif isinstance(value, list):
        #     if len(value) > 0 and isinstance(value[0], (float, int)):
        #         return np.array(value)
        #     else:
        #         return value
        # else:
        #     return value

    def __setitem__(self, key, value):
        self.storage[key] = value
        if not self.raw:
            self.storage = self._numpize_storage(self.storage)

    def __contains__(self, key):
        if key.startswith("/"):
            key = key[1:]
        return key in self.storage

    def create_dataset(self, key, data):
        self[key] = data

    def create_group(self, key):
        group = JSONStorageBackend(None, raw=True)
        self.storage[key] = {}
        group.storage = self.storage[key]
        return group

    def _numpize_storage(self, storage):
        if isinstance(storage, dict):
            return {k: self._numpize_storage(v) for k, v in storage.items()}
        if isinstance(storage, list):
            if len(storage) > 0 and isinstance(storage[0], str):
                return np.array(storage, dtype=bytes)
            return np.array(storage)
        if isinstance(storage, np.ndarray):
            return storage
        if isinstance(storage, str):
            return storage
        if isinstance(storage, (int, float)):
            return np.array(storage)
        return storage

    def _encode_storage(self, storage):
        if isinstance(storage, dict):
            return {k: self._encode_storage(v) for k, v in storage.items()}
        if isinstance(storage, list):
            return [self._encode_storage(v) for v in storage]
        if isinstance(storage, np.ndarray):
            if storage.dtype in (np.int, np.float32, np.float64):
                return storage.tolist()
            else:
                return [self._encode_storage(v) for v in storage]
        if isinstance(storage, str):
            return storage
        if isinstance(storage, bytes):
            return storage.decode("ascii")
        if isinstance(storage, (int, float)):
            return storage
        return storage

    # def _decode_storage(self, storage):
    #     if isinstance(storage, dict):
    #         return {k: self._decode_storage(v) for k, v in storage.items()}
    #     if isinstance(storage, list):
    #         return np.array(storage)
    #     return storage
