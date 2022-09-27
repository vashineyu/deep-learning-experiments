import os
import typing as t
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model as load_tfk_model
from tensorflow.python.keras.utils import multi_gpu_model as to_multi_gpu

from .inference.config import get_cfg_defaults
from .inference.utils import make_single_rendering_dict, update_json_dictionary
from .train.dataloader import GetDataset
from .train.model import preproc_resnet as preproc_fn
from .train.utils import check_cfg


def main():
    """
    Args:
        cfg: configuration yaml object
        model: model
    Returns: dictionary
    """
    def flat_items(d, key_separator='.'):
        for k, v in d.items():
            if type(v) is dict:
                for k1, v1 in flat_items(v, key_separator=key_separator):
                    yield key_separator.join((k, k1)), v1
            else:
                yield k, v

    test_config = {
        "SYSTEM.DEVICE": [],
        "SOURCE.TARGET_FILES": ["/mnt/nas/testcase_data/natural_image_01.jpg",
                                "https://cdn1.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg"],
        "MODEL.ROOT_PATH": "/data/seanyu/test/catdog_test",
        "SOURCE.RESULT_DIR": "/data/seanyu/test/catdog_test/outputs"
    }

    cfg_object = list(flat_items(test_config))
    cfg_object = [j for i in cfg_object for j in list(i)]

    cfg = get_cfg_defaults()
    if len(cfg_object) != 0:
        cfg.merge_from_list(cfg_object)
    cfg.freeze()
    check_cfg(cfg)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = ",".join(str(i) for i in cfg.SYSTEM.DEVICE)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    model_root = Path(cfg.MODEL.ROOT_PATH)
    model_path = model_root / "model.h5"
    model = load_model(str(model_path))

    """
    Create inference dataset
    """
    files_to_predict = {
        "file_to_predict": cfg.SOURCE.TARGET_FILES,
    }
    dataset = GetDataset(
        datapath_map=files_to_predict,
        classid_map=None,
        preproc_fn=preproc_fn,
        augment_fn=None,
        image_size=cfg.DATASET.IMAGE_SIZE
    )
    inference_array = np.array([
        next(dataset) for _ in range(len(dataset))
    ])

    y_pred = model.predict(inference_array, verbose=True)

    classmap = dict(cfg.DATASET.TARGET_REFERENCE)
    result_dir = Path(cfg.SOURCE.RESULT_DIR)
    result_dir.mkdir(exist_ok=True, parents=True)
    json_filepath = str(result_dir / "mapping.json")
    for index in range(len(cfg.SOURCE.TARGET_FILES)):

        json_item = make_single_rendering_dict(
            filename=cfg.SOURCE.TARGET_FILES[index],
            pred_array=y_pred[index],
            class_reference_table=classmap,
            path_to_cam="",
        )
        json_item = update_json_dictionary(
            json_filepath,
            json_item,
        )
        print(json_item)


def load_model(model_path: t.Union[str, Path]):
    """Load model
    Args:
        model_path: path to model
    Returns:
        model: model object
    """
    gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = 0 if gpu_devices is None else len(gpu_devices.split(","))
    model = load_tfk_model(model_path)
    if num_gpus > 1:
        model = to_multi_gpu(model, gpus=num_gpus, cpu_relocation=False, cpu_merge=False)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr=1e-8),
        )
    return model


if __name__ == "__main__":
    main()
