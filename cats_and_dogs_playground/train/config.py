from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICES = []

_C.SOURCE = CN()
_C.SOURCE.RESULT_DIR = ""

_C.DATASET = CN()
_C.DATASET.TRAIN_DIR = ["/mnt/extension/experiment/cat_dog/train/training"]
_C.DATASET.VALID_DIR = ["/mnt/extension/experiment/cat_dog/train/valid"]
_C.DATASET.TRAIN_RATIO = 0.9
_C.DATASET.IMAGE_SIZE = (256, 256, 3)
_C.DATASET.TARGET_REFERENCE = [
    ("cat", 0),
    ("dog", 1)
]
_C.DATASET.NUM_VALID_SIZE = 5000

_C.MODEL = CN()
_C.MODEL.BACKBONE = "R-50-v2"
_C.MODEL.BATCH_SIZE = 32
_C.MODEL.NUM_UPDATES = 500
_C.MODEL.EPOCHS = 50
_C.MODEL.LEARNING_RATE = 1e-4
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE = "bn"
_C.MODEL.OPTIMIZER = "Adam"


def get_cfg_defaults():
    return _C.clone()
