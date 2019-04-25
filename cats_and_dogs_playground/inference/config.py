# inference_defaults.py
from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = []
_C.SYSTEM.NUM_WORKERS = 4 # per gpu

_C.MODEL = CN()
# PATH to the model.h5 directory
_C.MODEL.ROOT_PATH = ""

_C.SOURCE = CN()
# Where to store results (such as overlay_alpha map, mapping.json. etc)
_C.SOURCE.RESULT_DIR = ""
# Files to run
_C.SOURCE.TARGET_FILES = [
    "/mnt/nas/testcase_data/natural_image_01.jpg", # cat
    "/mnt/nas/testcase_data/natural_image_02.jpg", # cat
    "/mnt/nas/testcase_data/natural_image_03.jpg", # cat
]
_C.SOURCE.EXTENSION = ".jpg"

_C.DATASET = CN()
# PATCH/TILE size
_C.DATASET.IMAGE_SIZE = (256, 256, 3)
_C.DATASET.TARGET_REFERENCE = [
    (0, "cat"),
    (1, "dog"),
]

_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 512 # per gpu
_C.INFERENCE.SAVE_JSON = 0 # True should be 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
