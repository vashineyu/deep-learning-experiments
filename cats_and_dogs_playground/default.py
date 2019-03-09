from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.NUM_WORKERS = 2
_C.SYSTEM.QUEUE_SIZE = 50
_C.SYSTEM.GPU_ID = 0
_C.SYSTEM.RESULT_DIR = "./results/"
_C.SYSTEM.NAME_FLAG = "default"
_C.SYSTEM.BACKBONE_PATH = ""

_C.DATASET = CN()
_C.DATASET.TRAIN = "/data/seanyu/cat_dog/dataset/train/"
_C.DATASET.TEST = "/data/seanyu/cat_dog/dataset/test1/"

_C.TRAIN = CN()
_C.TRAIN.TRAIN_RATIO = 0.9
_C.TRAIN.IMAGE_SIZE = (256, 256, 3)
_C.TRAIN.NUM_CLASSES = 2
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_UPDATES = 2000
_C.TRAIN.EPOCHS = 50
_C.TRAIN.LR = 1e-4
_C.TRAIN.USE_RESNET_PREPROC = True

_C.MODEL = CN()
_C.MODEL.BACKBONE = "R-50-v1" # R-50-v1, R-50-v2, R-50-xt
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE = "bn" # bn, gn
_C.MODEL.OPTIMIZER = "SGD" # SGD, Adam

_C.EXPERIMENT = CN()
_C.EXPERIMENT.ABC = 1
_C.EXPERIMENT.DEF = CN()
_C.EXPERIMENT.DEF.RGB = "test"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()