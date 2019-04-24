from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.DEVICES = []

_C.SOURCE = CN()
_C.SOURCE.RESULT_DIR = "" # Full path to store the result

_C.DATASET = CN()
_C.DATASET.TRAIN_DIR = ["/mnt/dataset/experiment/cat_dog/train/training"]
_C.DATASET.VALID_DIR = ["/mnt/dataset/experiment/cat_dog/train/valid"]
_C.DATASET.TRAIN_RATIO = 0.9 # if VALID_DIR is [], then it works
_C.DATASET.IMAGE_SIZE = (256, 256, 3)
_C.DATASET.TARGET_REFERENCE = [
    ("cat", 0),
    ("dog", 1)
]
_C.DATASET.NUM_VALID_SIZE = 5000

_C.MODEL = CN()
_C.MODEL.BACKBONE = "R-50-v2" # R-50-v1, R-50-v2, R-50-xt
_C.MODEL.BATCH_SIZE = 32
_C.MODEL.NUM_UPDATES = 500
_C.MODEL.EPOCHS = 50
_C.MODEL.LEARNING_RATE = 1e-4
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE = "bn" # bn, gn
_C.MODEL.OPTIMIZER = "Adam" # SGD, Adam

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()