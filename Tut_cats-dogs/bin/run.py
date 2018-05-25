from __future__ import print_function

import os
import glob
import re
import argparse
import pandas as pd
import numpy as np

from func.callbacks import *
from func.data_loader import *
from func.model import build_model
from func.TF_HandyTrainer import create_model

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tqdm import tqdm # if use notebook

from threading import Thread, Event
import queue

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import random

### Define arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=6)
parser.add_argument('--image_dir', default="/home/seanyu/datasets/cat_dog/dataset/")
parser.add_argument('--save_dir', default='./result')
parser.add_argument('--is_training', default=1, type=int)
parser.add_argument('--batch_size', default=48, type=int)
parser.add_argument('--do_augment', default=True, type = bool)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--image_size', default=(120,120,3), type = int)
parser.add_argument('--n_classes', default=2, type = int)
parser.add_argument('--train_ratio', default=0.9, type = float)
parser.add_argument('--use_model_ckpt', default = None, type = str)
parser.add_argument('--model_file_name', default = 'tmp')
parser.add_argument('--n_threads', default = 4, type = int)
parser.add_argument('--dq_size', default = 10, type = int)
parser.add_argument('--optimizer', default = 'sgd', type = str)
FLAGS = parser.parse_args()
print(FLAGS)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet

# Check and build avaiable path
if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

model_dir = FLAGS.save_dir + '/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

graphs_dir = FLAGS.save_dir + '/graphs'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

"""  
< Get data > 
Simply process data here
"""

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

"""
Data Augmentation setting
"""
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    sometimes(iaa.Affine(
            scale = (0.8,1.2),
            translate_percent = (-0.2, 0.2),
            rotate = (-20, 20),
            order = [0, 1],
            #cval = (0,255),
            mode = 'wrap'
            ))
])

# Basic Image reader, preprocess of images can placed here
def cv_load_and_resize(x, is_training = True):
    im_w, im_h, im_c = FLAGS.image_size
    im = cv2.imread(x)
    im = cv2.resize(im, (im_w, im_h))
    if FLAGS.do_augment and is_training:
        im = seq.augment_image(im)
    return im

data_gen = create_data_generator(df=[df_train_0, df_train_1], 
                                 n_classes = FLAGS.n_classes,
                                 image_size = FLAGS.image_size,
                                 do_augment = FLAGS.do_augment,
                                 aug_params=seq,
                                 batch_size=FLAGS.batch_size,
                                 n_batch = FLAGS.n_batch,
                                 open_image_handler=cv_load_and_resize)

data_gen.start_train_threads(FLAGS.n_threads, dq_size = FLAGS.dq_size)
train_gen = data_gen.get_data()

print("Loading evaluation data")
x_val, y_val = data_gen.get_evaluate_data(df_val)
print(x_val.shape)

# Call model
model_ops, metric_history = build_model(FLAGS)

# Set callbacks
cb_dict = {
    'reduce_lr' : ReduceLROnPlateau(lr=FLAGS.lr, factor=0.5, patience=3),
    'earlystop' : EarlyStopping(min_delta = 1e-4, patience= 15),
    'checkpoint' : Model_checkpoint(model_name=model_dir + '/' +  FLAGS.model_file_name, 
                                    save_best_only=True),
}
callback_dict = {
    'on_session_begin':[], # start of a session
    'on_batch_begin':[], # start of a training batch
    'on_batch_end':[], # end of a training batch
    'on_epoch_begin':[], # start of a epoch
    'on_epoch_end':[cb_dict['earlystop'], 
                    cb_dict['reduce_lr'],
                    cb_dict['checkpoint']], # end of a epoch
    'on_session_end':[] # end of a session
    }

callback_manager = Run_collected_functions(callback_dict)

# Train
trainer = TF_HandyTrainer(FLAGS=FLAGS, # hyper-parameters
                          data_gen=data_gen, # data generator
                          data_gen_get_data = train_gen,
                          model_ops=model_ops, # model
                          metric_history=metric_history, # metric recording
                          callback_manager=callback_manager# runable callbacks
                         )

trainer.initalize()
trainer.restore(model_to_restore='resnet_v2_50.ckpt', partial_restore=True)
trainer.do_training(cb_dict=cb_dict, validation_set=(x_val, y_val))

final_result = trainer.predict(x_val, model_to_restore=cb_dict['checkpoint'].model_name + '.ckpt')