import json
import os
import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt


def _init_json_file(json_filepath: t.Union[str, Path], refresh: bool = False):
    def create_file():
        with open(json_filepath, 'w') as f:
            json.dump({}, f)

    if os.path.exists(json_filepath):
        if refresh:
            create_file()
    else:
        create_file()


def update_json_dictionary(json_filepath: t.Union[str, Path], item: dict[str, t.Any]):
    try:
        with open(json_filepath, 'r') as f:
            json_object = json.load(f)

    except FileNotFoundError:
        _init_json_file(json_filepath)
        json_object = {}

    json_object.update(item)
    with open(json_filepath, 'w') as f:
        json.dump(json_object, f)

    return json_object


def make_single_rendering_dict(
    filename: t.Union[str, Path],
    pred_array: npt.NDArray,
    class_reference_table: list,
    path_to_cam: str = '',
):
    category = Path(filename).stem.split('.')[0]
    x = ','.join("{:4f}".format(i) for i in pred_array)

    pred_array_max = np.argmax(pred_array)
    item = {
        category: {
            "image_path": filename,
            "pred_class": class_reference_table[pred_array_max],
            "pred_value": str(pred_array[pred_array_max]),
            "predictions": x,
            "cam_path": path_to_cam,
        }
    }
    return item
