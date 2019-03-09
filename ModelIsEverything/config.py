# config.py
from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.NUM_WORKERS = 2
_C.SYSTEM.QUEUE_SIZE = 10
_C.SYSTEM.DEVICE = ""


_C.TRAIN = CN()
_C.TRAIN.NUM_CLASSES = 10
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.EPOCHS = 20
_C.TRAIN.LR = 1e-4


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()