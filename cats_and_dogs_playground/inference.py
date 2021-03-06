# inference.py

### import stuffs ###
import argparse
import os
from tensorflow.python.keras.models import load_model as load_tfk_model
from tensorflow.python.keras.utils import multi_gpu_model as to_multi_gpu
import tensorflow as tf
import numpy as np

from train.dataloader import GetDataset
from train.model import preproc_resnet as preproc_fn
from train.utils import try_makedirs, check_cfg
from inference.config import get_cfg_defaults
from inference.utils import update_json_dictionary, make_single_rendering_dict
### Main functions ###
def run(cfg, model=None):
    """
    Args:
        cfg: configuration yaml object
        model: model
    Returns: dictionary
    """

    """
    Load Model
    """
    if model is None:
        devices = list_devices_to_string(cfg.SYSTEM.DEVICE)
        set_device(devices)
        model_path = os.path.join(cfg.MODEL.ROOT_PATH, "model.h5")
        model = load_model(model_path)

    """
    Create inference dataset
    """
    files_to_predict = {"file_to_predict":cfg.SOURCE.TARGET_FILES}
    dataset = GetDataset(datapath_map=files_to_predict,
                         classid_map=None,
                         preproc_fn=preproc_fn,
                         augment_fn=None,
                         image_size=cfg.DATASET.IMAGE_SIZE
                         )
    inference_array = np.array([next(dataset) for _ in range(len(dataset))])
    #print(inference_array.shape)

    y_pred = model.predict(inference_array, verbose=True)

    classmap = dict(cfg.DATASET.TARGET_REFERENCE)
    try_makedirs(cfg.SOURCE.RESULT_DIR)
    for index in range(len(cfg.SOURCE.TARGET_FILES)):
        """
        TO DO: Class Activation Map
        """

        """
        Generate outputs
        """
        json_filepath = os.path.join(cfg.SOURCE.RESULT_DIR, "mapping.json")

        json_item = make_single_rendering_dict(filename=cfg.SOURCE.TARGET_FILES[index],
                                               pred_array=y_pred[index],
                                               class_reference_table=classmap,
                                               path_to_cam=""
                                               )
        print(json_item)
        update_json_dictionary(json_filepath, json_item)



def list_devices_to_string(list_item):
    """Convert cfg devices into comma split format.
    Args:
        list_item (list): list of devices, e.g. [], [1], ["1"], [1,2], ...
    Returns:
        devices (string): comma split devices
    """
    return ",".join(str(i) for i in list_item)

def set_device(devices):
    """set os devices.
    Args:
        devices (str): comma split string
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    try:
        os.environ["CUDA_VISIBLE_DEVICES"]
    except KeyError:
        print("CUDA devices not set")
        raise
    return True

def count_devices(devices):
    """Count numbers of CUDA devices
    Args:
        devices (str): comma split string
    Returns:
        num_gpus (int): numbers of gpus
    """
    num_gpus = len(devices.split(","))
    return num_gpus

def load_model(model_path):
    """Load model
    Args:
        model_path: path to model
    Returns:
        model: model object
    """
    gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = count_devices(gpu_devices) if gpu_devices is not None else 0
    model = load_tfk_model(model_path)
    if num_gpus > 1:
        model = to_multi_gpu(model, gpus=num_gpus, cpu_relocation=False, cpu_merge=False)
        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(lr=1e-8))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cats/Dogs playground parameters")
    parser.add_argument(
        "--config-file",
        default=None,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    check_cfg(cfg)

    run(cfg)
