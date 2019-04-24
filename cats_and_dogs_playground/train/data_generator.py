import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd

import multiprocessing as mp
from multiprocessing import Event
import queue

### DONT CHANGE IF NOT TESTING FOR DATASET LOADER ###
class GetDataset():
    def __init__(self, df_list, class_id, n_classes, f_input_preproc, image_size=(256,256,3), onehot=True, augmentation=None):
        
        self.df_list = df_list
        self.class_id = class_id
        self.n_classes = n_classes
        self.preproc = f_input_preproc
        self.image_size = image_size
        self.onehot = onehot
        self.aug = augmentation
        
        ## Init ##
        self.df_list = self.df_list.sample(frac=1.).reset_index(drop=True)
        self.current_index = 0
    
    def __len__(self):
        return len(self.df_list)
    
    def __getitem__(self, idx):
        
        img = self.load_image(img_path=self.df_list.iloc[self.current_index]['img_path'], image_size=self.image_size)
        
        if self.aug is not None:
            img = self.aug.augment_image(img)
            
        img = img.astype(np.float32)
        
        if self.preproc is not None:
            img = self.preproc(img)
        
        label = self.df_list.iloc[self.current_index]['cate']
        if self.onehot:
             label = tf.keras.utils.to_categorical(label, num_classes=self.n_classes)
        
        self.current_index = (self.current_index + 1) % len(self.df_list)
        return img, label
    
    def __next__(self):
        return self.__getitem__(idx=self.current_index)
    
    @staticmethod
    def load_image(img_path, image_size):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size[0], image_size[1]))
        return img
    
class Customized_dataloader():
    """
    1. Compose multiple generators together
    2. Make this composed generator into multi-processing function
    """
    def __init__(self, list_dataset, batch_size_per_dataset=16, queue_size=128, num_workers=0):
        """
        Args:
            - list_dataset: put generator object as list [gen1, gen2, ...]
            - batch_size_per_dataset: bz for each generator (total_batch_size/n_class)
            - queue_size: queue size
            - num_workers: start n workers to get data
        
        Action: Call with next
        """
        self.list_dataset = list_dataset
        self.batch_size_per_dataset = batch_size_per_dataset
        self.sample_queue = mp.Queue(maxsize = queue_size)
        
        self.jobs = num_workers
        self.events = list()
        self.workers = list()
        for i in range(num_workers):
            event = Event()
            work = mp.Process(target = enqueue, args = (self.sample_queue, event, self.compose_data))
            work.daemon = True
            work.start()
            self.events.append(event)
            self.workers.append(work)
        print("workers ready")
        
    def __next__(self):
        return self.sample_queue.get()
    
    def compose_data(self):
        while True:
            imgs, labels = [], []
            for z in range(self.batch_size_per_dataset):
                data = [next(i) for i in self.list_dataset]
                img, label = zip(*data)
                imgs.append(np.array(img))
                labels.append(np.array(label))
            yield np.concatenate(imgs), np.concatenate(labels)
    
    def stop_worker(self):
        for t in self.events:
            t.set()
        for i, t in enumerate(self.workers):
            t.join(timeout = 1)
        print("all_worker_stop")

# ----- #
def enqueue(queue, stop, gen_func):
    gen = gen_func()
    while True:
        if stop.is_set():
            return
        queue.put(next(gen))