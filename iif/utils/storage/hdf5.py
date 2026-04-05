import datetime
import os
import time

from . import StorageBackend


class HDF5StorageBackend(StorageBackend):
    """
    Class handling the access to the dataset. Needs to be thread-safe
    """
    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        open_mode = "r" if self.readonly else "a"
        locking = not self.readonly

        start = datetime.datetime.now()
        while (datetime.datetime.now() - start).total_seconds() * 1000 < self.timeout:
            try:
                import h5py as h5py
                file = h5py.File(self.path, open_mode, locking=locking)
                return file
            except BlockingIOError:
                time.sleep(0.1)
        else:
            raise BlockingIOError(f"Could not open file in {self.timeout}ms")

    def close(self, storage):
        storage.close()

    def __enter__(self):
        self.storage = self.open()
        self.lock = True
        return self.storage

    def __exit__(self, type, value, traceback):
        self.close(self.storage)
        self.storage = None
        self.lock = None

    def __getitem__(self, key):
        return self.storage[f"/{key}"]

    def __setitem__(self, key, value):
        self.storage[f"/{key}"] = value
