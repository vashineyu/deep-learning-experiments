from __future__ import print_function
#
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
from time import sleep
from tqdm import tqdm # if use notebook

import multiprocessing as mp
from multiprocessing import Event
import queue
import sys

from PIL import Image
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import random
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=5)
parser.add_argument('--image_dir', default="/data/seanyu/cat_dog/dataset/")
parser.add_argument('--saving_flag', default=None)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_updates', default=2000, type=int)
parser.add_argument('--do_augment', default=True, type = bool)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--image_size', default=(256,256,3), type = int)
parser.add_argument('--n_classes', default=2, type = int)
parser.add_argument('--n_batch', default=100, type = int)
parser.add_argument('--train_ratio', default=0.9, type = float)
parser.add_argument('--norm_use', default="bn", type=str)
parser.add_argument('--model_file_name', default = 'model.h5')

sys.path.append("experimental_stuff/")
from backbone import *
from model import build_model

def main():
    FLAGS = parser.parse_args()
    print(FLAGS)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
    """  Get data """
    d_train = FLAGS.image_dir + '/train/'
    d_test = FLAGS.image_dir + '/test1/'
    image_train_list = glob.glob(d_train + '*.jpg')
    image_test_list = glob.glob(d_test + '*.jpg')

    df_train = pd.DataFrame({'img_path': image_train_list})
    df_test = pd.DataFrame({'img_path': image_test_list})

    df_train['cate'] = df_train.img_path.apply(os.path.basename)
    df_train['cate'] = [i.split(".")[0] for i in list(df_train.cate)]
    df_train.cate = df_train.cate.replace({'dog': 0, 'cat': 1})

    nb_epoch = FLAGS.epochs

    df_train_0, df_val_0 = train_test_split(df_train[df_train['cate'] == 0], test_size = 1-FLAGS.train_ratio)
    df_train_1, df_val_1 = train_test_split(df_train[df_train['cate'] == 1], test_size = 1-FLAGS.train_ratio)

    df_val = pd.concat((df_val_0, df_val_1)).reset_index(drop = True)

    del df_val_0, df_val_1
    
    USE_RESNET_PREPROC = True
    dtrain = GetDataset(df_list=df_train,
                           class_id=0, n_classes=2,
                           f_input_preproc=preproc if not USE_RESNET_PREPROC else tf.keras.applications.resnet50.preprocess_input,
                           augmentation=Augmentation_Setup.augmentation, 
                           onehot= True, 
                           image_size=FLAGS.image_size)
    dvalid = GetDataset(df_list=df_val, 
                           class_id=0, n_classes=2,
                           f_input_preproc=preproc if not USE_RESNET_PREPROC else tf.keras.applications.resnet50.preprocess_input,
                           augmentation=None, 
                           onehot= True, 
                           image_size=FLAGS.image_size)
    valid_gen = Customized_dataloader([dvalid], batch_size_per_dataset=1, num_workers=1)
    x_val, y_val = [], []
    for _ in tqdm(range(500)):
        a,b = next(valid_gen)
        x_val.append(a)
        y_val.append(b)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    valid_gen.stop_worker()

    print(x_val.shape)
    print(y_val.shape)
    print(y_val.sum(axis=0))
    
    train_gen = Customized_dataloader([dtrain], 
                                      batch_size_per_dataset=FLAGS.batch_size, 
                                      num_workers=4, queue_size=50)
    
    tf.keras.backend.clear_session()
    model = build_model(model_fn=ResNet50V2, norm_use=FLAGS.norm_use)
    
    optim = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    model.compile(loss='categorical_crossentropy', 
                  metrics=["accuracy"], 
                  optimizer=optim)
    model.summary()
    cb_list = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                patience=4,
                                                min_lr=1e-12),
          ]
    model.fit_generator(train_gen,
                        epochs=FLAGS.epochs,
                        steps_per_epoch=FLAGS.n_updates, 
                        validation_data=(x_val, y_val),
                        callbacks=cb_list
                        )
    
    train_loss = model.history.history['loss']
    valid_loss = model.history.history['val_loss']
    train_acc = model.history.history['acc']
    valid_acc = model.history.history['val_acc']

    if FLAGS.saving_flag is None:
        FLAGS.saving_flag = "no-master"

    plt.figure(figsize=(8,6))
    plt.plot(range(len(train_loss)), train_loss, label='train_loss')
    plt.plot(range(len(valid_loss)), valid_loss, label='valid_loss')
    plt.legend()
    plt.savefig(os.path.join("results", "exp_" + FLAGS.saving_flag + "_loss.png"))

    plt.figure(figsize=(8,6))
    plt.plot(range(len(train_acc)), train_acc, label='train_accuracy')
    plt.plot(range(len(valid_acc)), valid_acc, label='valid_accuracy')
    plt.legend()
    plt.savefig(os.path.join("results", "exp_" + FLAGS.saving_flag + "_acc.png"))
    
    result_df = pd.DataFrame({"train_loss":train_loss,
                              "valid_loss":valid_loss,
                              "train_acc":train_acc,
                              "valid_acc":valid_acc
                             })
    result_df.to_csv(os.path.join("results", "exp_" + FLAGS.saving_flag + "_result.csv"), index=False)
    print("All Done")
    


"""
def build_model_graph(model_fn, norm_use, input_shape=(256,256,3), n_outputs=2):
    pretrain_modules = model_fn(include_top=False, input_shape=input_shape, norm_use=norm_use, weights=None)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    out = tf.keras.layers.Dense(units=n_outputs, activation='softmax', name='output')(gap)
    model = tf.keras.models.Model(inputs=[pretrain_modules.input], outputs=[out])
    return model    
"""

class Augmentation_Setup(object):  
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    lesstimes = lambda aug: iaa.Sometimes(0.2, aug)
    
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5, name="FlipLR"),
        iaa.Flipud(0.5, name="FlipUD"),
        iaa.OneOf([iaa.Affine(rotate = 90),
                   iaa.Affine(rotate = 180),
                   iaa.Affine(rotate = 270)]),
        sometimes(iaa.Affine(
                    scale = (0.8,1.2),
                    translate_percent = (-0.2, 0.2),
                    rotate = (-15, 15),
                    mode = 'wrap'
                    ))
    ])

def preproc(img):
    #return (img - img.min()) / (img.max() - img.min())
    return img / 255.
    
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
    

if __name__ == '__main__':
    main()
