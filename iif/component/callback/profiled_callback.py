import time
from functools import partial

import pytorch_lightning as pl

from iif.utils.logging import init_logger


class ProfiledCallback(pl.Callback):
    def __init__(self, callback=None):
        self.module_logger = init_logger()
        self.callback = callback or self

    def profile_wrapper(self, fn_name, *args, **kwargs):
        start_time = time.time()

        fn = getattr(self.callback, fn_name)
        result = fn(*args, **kwargs)

        end_time = time.time()

        elapsed_sec = (end_time - start_time)
        self.module_logger.info(f"Call {self.callback.__class__.__name__}.{fn_name} took {self._sec_to_str(elapsed_sec)}")

        return result

    def _sec_to_str(self, elapsed_sec):
        if elapsed_sec < 1:
            return f"{elapsed_sec * 1e6} us"
        elif elapsed_sec < 60:
            return f"{elapsed_sec} s {(elapsed_sec % 1) * 1e6} us"
        else:
            return f"{elapsed_sec // 60} m {elapsed_sec % 60} s {(elapsed_sec % 1) * 1e6} us"

    def __getattr__(self, fn_name):
        if hasattr(self.callback, fn_name):
            return partial(self.profile_wrapper, fn_name=fn_name)
        else:
            raise AttributeError(f'No such field/method: {fn_name}')

    def __call__(self, *args, **kwargs):
        return partial(self.profile_wrapper, fn_name="__call__")(*args, **kwargs)

    def __repr__(self,):
        return f"ProfiledCallback({self.callback})"
