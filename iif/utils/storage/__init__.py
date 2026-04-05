import os
from abc import abstractmethod

from ..logging import init_logger


class StorageBackend:
    """
        Class handling the access to the dataset. Needs to be thread-safe
        """

    def __init__(self, path, required_fields=None, readonly=False, timeout=600000):
        self.module_logger = init_logger()

        self.path = path
        self.required_fields = required_fields
        self.readonly = readonly
        self.timeout = timeout

        if self.path is not None:
            self.lock_path = self.path + ".lock"
        self.lock = None

        self.storage = None

    def exists(self):
        if self.required_fields is not None:
            if not os.path.exists(self.path):
                return False
            else:
                with self as storage:
                    for required_field in self.required_fields:
                        if required_field not in storage:
                            self.module_logger.warning(f"Required field {required_field} is missing! Removing the current file!")
                            os.remove(self.path)
                            return False
                return True
        else:
            return os.path.exists(self.path)

    def __enter__(self):
        from filelock import FileLock

        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self.lock = FileLock(self.lock_path, timeout=self.timeout)
        self.lock.acquire()
        self.storage = self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close(self.storage)
        self.storage = None
        self.lock.release()
        self.lock = None

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self, storage):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

