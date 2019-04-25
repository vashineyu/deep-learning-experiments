import json
import os
import numpy as np

def _init_json_file(jfile, refresh=False):
    def create_file():
        with open(jfile, 'w') as f:
            json.dump({}, f)

    if os.path.exists(jfile):
        if refresh:
            create_file()
    else:
        create_file()

def update_json_dictionary(jfile, item):
    try:
        with open(jfile, 'r') as f:
            json_object = json.load(f)
    except FileNotFoundError:
        _init_json_file(jfile)
        json_object = {}

    json_object.update(item)
    with open(jfile, 'w') as f:
        json.dump(json_object, f)

    return json_object

def make_single_rendering_dict(filename,
                               pred_array,
                               class_reference_table,
                               path_to_cam=''):
    keyname = os.path.basename(filename).split(".")[:-1][0]
    x = ','.join("{:4f}".format(i) for i in pred_array)

    #assert len(pred_array) == 1, "Length of pred_array should be 1"
    pred_array_max = np.argmax(pred_array)
    item = {
        keyname:{
            "image_path":filename,
            "pred_class":class_reference_table[pred_array_max],
            "pred_value":str(pred_array[pred_array_max]),
            "predictions":x,
            "cam_path":path_to_cam
        }
    }
    return item