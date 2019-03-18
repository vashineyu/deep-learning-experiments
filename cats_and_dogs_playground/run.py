from __future__ import print_function
import argparse
import cv2
import glob
import os
import re
import random
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Cats/Dogs playground")
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

def main():
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    sys.path.append(cfg.SYSTEM.BACKBONE_PATH)
    print(cfg)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    
    """  Get data """
    image_train_list = glob.glob(cfg.DATASET.TRAIN + '*.jpg')
    image_test_list = glob.glob(cfg.DATASET.TEST + '*.jpg')

    df_train = pd.DataFrame({'img_path': image_train_list})
    

    df_train['cate'] = df_train.img_path.apply(os.path.basename)
    df_train['cate'] = [i.split(".")[0] for i in list(df_train.cate)]
    df_train.cate = df_train.cate.replace({'dog':0, 'cat':1})

    _, df_val_0 = train_test_split(df_train[df_train['cate'] == 0], test_size=1-cfg.TRAIN.TRAIN_RATIO)
    _, df_val_1 = train_test_split(df_train[df_train['cate'] == 1], test_size=1-cfg.TRAIN.TRAIN_RATIO)
    df_val = pd.concat((df_val_0, df_val_1)).reset_index(drop=True)

    del df_val_0, df_val_1
    
    USE_RESNET_PREPROC = cfg.TRAIN.USE_RESNET_PREPROC
    dtrain = GetDataset(df_list=df_train,
                        class_id=0, n_classes=2,
                        f_input_preproc=preproc if not USE_RESNET_PREPROC else tf.keras.applications.resnet50.preprocess_input,
                        augmentation=Augmentation_Setup.augmentation, 
                        onehot=True, 
                        image_size=cfg.TRAIN.IMAGE_SIZE)
    dvalid = GetDataset(df_list=df_val, 
                        class_id=0, n_classes=2,
                        f_input_preproc=preproc if not USE_RESNET_PREPROC else tf.keras.applications.resnet50.preprocess_input,
                        augmentation=None, 
                        onehot=True, 
                        image_size=cfg.TRAIN.IMAGE_SIZE)
    
    valid_gen = Customized_dataloader([dvalid], batch_size_per_dataset=16, num_workers=1)
    x_val, y_val = [], []
    for _ in tqdm(range(100)):
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
                                      batch_size_per_dataset=cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_CLASSES, 
                                      num_workers=cfg.SYSTEM.NUM_WORKERS, 
                                      queue_size=cfg.SYSTEM.QUEUE_SIZE)
    
    
    model = build_model(backbone=cfg.MODEL.BACKBONE, 
                        norm_use=cfg.MODEL.NORM_USE, weights="imagenet" if cfg.MODEL.USE_PRETRAIN else None)
    optim = make_optimizer(cfg)
    
    model.compile(loss='categorical_crossentropy', 
                  metrics=["accuracy"], 
                  optimizer=optim)
    model.summary()
    cb_list = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                patience=4,
                                                min_lr=1e-12),
          ]
    model.fit_generator(train_gen,
                        epochs=cfg.TRAIN.EPOCHS,
                        steps_per_epoch=cfg.TRAIN.NUM_UPDATES, 
                        validation_data=(x_val, y_val),
                        callbacks=cb_list
                        )
    
    train_loss = model.history.history['loss']
    valid_loss = model.history.history['val_loss']
    train_acc = model.history.history['acc']
    valid_acc = model.history.history['val_acc']

    plt.figure(figsize=(8,6))
    plt.plot(range(len(train_loss)), train_loss, label='train_loss')
    plt.plot(range(len(valid_loss)), valid_loss, label='valid_loss')
    plt.legend()
    plt.savefig(os.path.join("results", "exp_" + cfg.SYSTEM.NAME_FLAG + "_loss.png"))

    plt.figure(figsize=(8,6))
    plt.plot(range(len(train_acc)), train_acc, label='train_accuracy')
    plt.plot(range(len(valid_acc)), valid_acc, label='valid_accuracy')
    plt.legend()
    plt.savefig(os.path.join("results", "exp_" + cfg.SYSTEM.NAME_FLAG + "_acc.png"))
    
    result_df = pd.DataFrame({"train_loss":train_loss,
                              "valid_loss":valid_loss,
                              "train_acc":train_acc,
                              "valid_acc":valid_acc
                             })
    result_df.to_csv(os.path.join("results", "exp_" + cfg.SYSTEM.NAME_FLAG + "_result.csv"), index=False)
    print("All Done")
    
class Augmentation_Setup(object):  
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    lesstimes = lambda aug: iaa.Sometimes(0.2, aug)
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5, name="FlipLR"),
        iaa.Flipud(0.5, name="FlipUD"),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        sometimes(iaa.Affine(
                    scale=(0.8,1.2),
                    translate_percent=(-0.2, 0.2),
                    rotate=(-15, 15),
                    mode="wrap"
                    ))
    ])
    
if __name__ == '__main__':
    main()
