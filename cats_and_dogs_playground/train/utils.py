"""utils.py
Related util functions
"""
import os
import glob

def check_cfg(cfg):
    if cfg.SOURCE.RESULT_DIR == "":
        assert False, "SOURCE.RESULT_DIR should not be empty string"

    return True

def try_makedirs(path):
    try:
        os.makedirs(path)
    except:
        pass
    return True

def fetch_path_from_dirs(list_of_search_dirs, key, image_format="jpg"):
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
        #this_search_path = os.path.join(d, "*"+key+"*"+image_format)
        outputs.extend(glob.glob(this_search_path))
    return outputs