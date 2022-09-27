"""utils.py
Related util functions
"""
import glob
import os
import time
import typing as t

from tensorflow.python.keras.callbacks import Callback


def check_cfg(cfg):
    if cfg.SOURCE.RESULT_DIR == "":
        ValueError("SOURCE.RESULT_DIR should not be empty string")
    return True


def fetch_path_from_dirs(list_of_search_dirs, key):
    """list all files in list of dirs

    Args:
        list_of_search_dirs:
        key: specific search key, e.g. cat/dog/...

    Returns:
        list of image path

    """
    outputs = []
    for d in list_of_search_dirs:
        this_search_path = os.path.join(d, "*" + key + "*")
        outputs.extend(glob.glob(this_search_path))
    return outputs


class Timer(Callback):
    """Time recording

    record_batch_per_period: period for recording batch time, default=1. If steps

    Usage:
    model_timer = Timer()
    callbacks.append(model_timer)
    ...afer training...
    model_timer.timer --> dict of time recording
    """

    def __init__(self, record_batch_per_period: int = 1):
        self.record_batch_per_period = record_batch_per_period

    def on_train_begin(self, logs={}):
        self.timer = {
            "train_start": time.time(),
            "train_end": -1,
            "epoch_start": [],
            "epoch_end": [],
            "batch_start": [],
            "batch_end": [],
        }

    def on_train_end(self, logs={}):
        self.timer["train_end"] = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.timer['epoch_start'].append(time.time())

    def on_epoch_end(self, epoch, logs={}):
        self.timer['epoch_end'].append(time.time())

    def on_batch_begin(self, batch, logs={}):
        if batch % self.record_batch_per_period == 0:
            self.timer["batch_start"].append(time.time())

    def on_batch_end(self, batch, logs={}):
        if batch % self.record_batch_per_period == 0:
            self.timer["batch_end"].append(time.time())
