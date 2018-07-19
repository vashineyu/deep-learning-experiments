import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
import random

def cv_load_and_resize(x, image_size, is_training = True, do_augment = False, seq = None):
    im_w, im_h, im_c = image_size
    im = cv2.imread(x)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape != image_size:
        im = cv2.resize(im, (im_w, im_h))
    if do_augment and is_training:
        im = seq.augment_image(im)
    return im

def cv_load_and_pad(x, image_size, is_training = True, do_augment = False, seq = None):
    im_w, im_h, im_c = image_size
    im = cv2.imread(x)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if im.shape != image_size:
        im = image_crop_and_pad(im, im_w) # crop and pad to certain size
    if do_augment and is_training:
        im = seq.augment_image(im)
    return im

def image_crop_and_pad(im, target_size):
    # target size: should be a single value
    
    im_h, im_w, *_ = im.shape
    
    lr_pad = (target_size - (im_w-1) )// 2 # minus 1 additionally, prevent odd value that pad not enough
    tb_pad = (target_size - (im_h-1) )// 2 
    
    # Check estimated padding larger than 0
    lr_pad = 0 if lr_pad < 0 else lr_pad
    tb_pad = 0 if tb_pad < 0 else tb_pad
    im = cv2.copyMakeBorder(im, tb_pad, tb_pad, lr_pad, lr_pad, cv2.BORDER_REFLECT) # WRAP, REPLICATE, REFLECT
    # random crop
    new_h, new_w, *_ = im.shape
    h_rand = 0 if new_h == target_size else np.random.randint(new_h - target_size)
    w_rand = 0 if new_w == target_size else np.random.randint(new_w - target_size)
    
    im = im[h_rand:(h_rand+target_size), w_rand:(w_rand+target_size), ...]
    # fix crop
    #im = im[:target_size, :target_size, ...]
    
    return im
