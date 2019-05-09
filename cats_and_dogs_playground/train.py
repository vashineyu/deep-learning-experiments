"""train.py
main function to train the cat/dog classifier
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from train.config import get_cfg_defaults
from train.augment import Augmentation_Setup as augment_fn
from train.dataloader import GetDataset, DataLoader
from train.model import build_model, make_optimizer
from train.model import preproc_resnet as preproc_fn
from train.utils import check_cfg, try_makedirs, fetch_path_from_dirs, Timer

import horovod.tensorflow.keras as hvd
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


device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = device

"""
# Hovovod Setting
"""
hvd.init()
tf_config = tf.ConfigProto()

tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.visible_device_list = str(hvd.local_rank())
session = tf.Session(config=tf_config)
tf.keras.backend.set_session(session)

"""  Get data """
dict_target = dict(cfg.DATASET.TARGET_REFERENCE)
dict_image_train = {}
dict_image_valid = {}
for key in dict_target.keys():
    dict_image_train[key] = fetch_path_from_dirs(cfg.DATASET.TRAIN_DIR, key=key)
    if len(cfg.DATASET.VALID_DIR) == 0:
        dict_image_train[key], dict_image_valid[key] = train_test_split(dict_image_train[key], test_size=(1-cfg.DATASET.TRAIN_RATIO))
    else:
        dict_image_valid[key] = fetch_path_from_dirs(cfg.DATASET.VALID_DIR, key=key)

dataset_train = GetDataset(datapath_map=dict_image_train,
                           classid_map=dict_target,
                           preproc_fn=preproc_fn,
                           augment_fn=augment_fn.augmentation,
                           image_size=cfg.DATASET.IMAGE_SIZE)
dataset_valid = GetDataset(datapath_map=dict_image_valid,
                           classid_map=dict_target,
                           preproc_fn=preproc_fn,
                           augment_fn=None,
                           image_size=cfg.DATASET.IMAGE_SIZE)
dataloader = DataLoader(dataset=dataset_train,
                        num_classes=len(dict_target),
                        batch_size=cfg.MODEL.BATCH_SIZE)

x_val, y_val = [], []
for _ in tqdm(range(cfg.DATASET.NUM_VALID_SIZE)):
    a,b = next(dataset_valid)
    x_val.append(a)
    y_val.append(b)
x_val = np.array(x_val)
y_val = np.array(y_val)

print("Validation input size: %s" % len(x_val))
print("Validation output size: %s" % y_val.shape[-1])
print(y_val.sum(axis=0))


model = build_model(input_shape=cfg.DATASET.IMAGE_SIZE,
                    num_classes=len(dict_target),
                    backbone=cfg.MODEL.BACKBONE, 
                    norm_use=cfg.MODEL.NORM_USE,
                    weights="imagenet" if cfg.MODEL.USE_PRETRAIN else None)
optim = make_optimizer(optimizer=cfg.MODEL.OPTIMIZER, learning_rate=cfg.MODEL.LEARNING_RATE)
optim = hvd.DistributedOptimizer(optim)

model.compile(loss='categorical_crossentropy', 
              metrics=["accuracy"], 
              optimizer=optim)
model.summary()
cb_list = []

"""
# Horovod Setting
"""
cb_list.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
cb_list.append(hvd.callbacks.MetricAverageCallback())
cb_list.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

cb_list.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-12))
model_timer = Timer(record_batch_per_period=100)
cb_list.append(model_timer)


if hvd.rank() == 0:
    cb_list.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(cfg.SOURCE.RESULT_DIR, "model.h5"),
                                                      monitor="val_loss",
                                                      save_best_only=True))
    cb_list.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.SOURCE.RESULT_DIR, "logs")))

try_makedirs(cfg.SOURCE.RESULT_DIR)
model.fit_generator(dataloader,
                    epochs=cfg.MODEL.EPOCHS,
                    steps_per_epoch=len(dataloader) if cfg.MODEL.NUM_UPDATES == 0 else cfg.MODEL.NUM_UPDATES,
                    validation_data=(x_val, y_val),
                    callbacks=cb_list
                    )

train_loss = model.history.history['loss']
valid_loss = model.history.history['val_loss']
train_acc = model.history.history['acc']
valid_acc = model.history.history['val_acc']
result_df = pd.DataFrame({"train_loss":train_loss,
                          "valid_loss":valid_loss,
                          "train_acc":train_acc,
                          "valid_acc":valid_acc
                         })
result_df.to_csv(os.path.join(cfg.SOURCE.RESULT_DIR, "valid_result.csv"), index=False)
print(model_timer.timer)
print("All Done")

if hvd.rank() == 0:
    import time
    time.sleep(5)