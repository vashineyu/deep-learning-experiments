import cv2
import os
import requests
import numpy as np
from random import shuffle
from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

class GetDataset():
    def __init__(self,
                 datapath_map,
                 classid_map,
                 preproc_fn=None,
                 augment_fn=None,
                 image_size=(256,256,3),
                 do_shuffle=True):
        """Initalize dataset object

        Args:
            datapath_map (dict): dictionary of class with classname, {"cat":["c1.png", "c2.png"], "dog":["d1.png", "d2.png"]}
            classid_map (dict): dictionary of class with target id, {"cat": 0, "dog": 1}
            preproc_fn (function): preproc function
            augment_fn (function): augment function
            image_size (tuple): tuple of image_size, (256, 256)
            do_shuffle (bool): do shuffle data every iteration over dataset
        """
        self.datapath_map = datapath_map
        self.class_map = classid_map
        self.preproc_fn = preproc_fn
        self.augment_fn = augment_fn
        self.image_size = image_size
        self.do_shuffle = do_shuffle

        ## Init ##
        self.datalist = [item for key in datapath_map for item in datapath_map[key]]
        self.classlist = [key for key in datapath_map for _ in datapath_map[key]]
        self.current_index = 0
        self._shuffle_item()

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        w, h, *_ = self.image_size
        image = self.load_image(img_path=self.datalist[idx], image_size=(w, h))
        label = to_categorical(self.class_map[self.classlist[idx]], len(self.class_map))
        
        if self.augment_fn is not None:
            image = self.augment_fn.augment_image(image)
        image = image.astype(np.float32)
        
        if self.preproc_fn is not None:
            image = self.preproc_fn(image)

        self.current_index = (self.current_index + 1) % len(self.datalist)
        if (self.current_index == 0) & (self.do_shuffle):
            self._shuffle_item()

        return image, label
    
    def __next__(self):
        return self.__getitem__(idx=self.current_index)

    def _shuffle_item(self):
        temp_item = list(zip(self.datalist, self.classlist))
        shuffle(temp_item)
        self.datalist, self.classlist = zip(*temp_item)
    
    @staticmethod
    def load_image(img_path, image_size):
        if "http" in img_path:
            img = url2img(img_path)
        else:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size[0], image_size[1]))
        return img

def url2img(url):
    """
    Args:
        url: url path to a image

    Returns:
        img: image array

    """
    resp = requests.get(url)
    img = cv2.imdecode(np.asarray(bytearray(resp.content), dtype='uint8'), -1)
    return img

class DataLoader(Sequence):
    """
    1. Compose multiple generators together
    2. Make this composed generator into multi-processing function
    """
    def __init__(self, dataset, num_classes, batch_size=32, print_recording_period=5000):
        """

        Args:
            dataset:
            num_classes:
            batch_size:
            print_recording_period:
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.print_recording_period = print_recording_period
        
        self.y_counter = np.array([0.] * num_classes)
        
    def __len__(self):
        """
        Returns: len of object
        """
        return len(self.dataset) // self.batch_size
    
    def __getitem__(self, index):
        xs = []
        ys = []
        for i in range(self.batch_size):
            img, gt = next(self.dataset)
            xs.append(img)
            ys.append(gt)
        xs = np.array(xs)
        ys = np.array(ys)

        self.y_counter += ys.sum(axis=0)
        if (index % self.print_recording_period) == 0:
            print(" || Numbers of class being sampled: %s" % (self.y_counter))
        return xs, ys
