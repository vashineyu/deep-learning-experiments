## python utils

import cv2
import os
import glob
import pandas as pd
import numpy as np

# --- 

# The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

### encode label functions 
def label_reconstruct(x):
    tmp = x.split(" ")
    start_point = tmp[0::2]
    expand_len = tmp[1::2]
    mask = []
    for i,j in zip(start_point, expand_len):
        cur = list(range(int(i), int(i) + int(j)))
        mask.extend(cur)
    return mask

def label_to_mask(x, im_size):
    im_w, im_h, *_ = im_size # if we pass only 2 index tuple, it still works?
    #x = [ix - im_h - 1 for ix in x] # make the first pixel1 as (0,0)
    x = [ix - 1 for ix in x]
    
    # background
    background = np.ones((im_w * im_h))
    background[x] = 0
    background = background.reshape((im_w, im_h), order = "F")
    
    # foreground
    mask = np.zeros((im_w * im_h))
    mask[x] = 1
    mask = mask.reshape((im_w, im_h), order = "F")
    
    images = np.dstack((background, mask))
    
    return images


def onehot_to_mask(x, input_shape, output_shape, n_classes = 2):
    oim_w, oim_h = input_shape
    oim_c = n_classes
    
    tim_w, tim_h, tim_c = output_shape
    
    x = x.reshape((oim_w, oim_h, oim_c))
    x = cv2.resize(x, (tim_h, tim_w))
    x = x.argmax(axis = -1)
    return x

def mask_to_label(x):
    # the input should be a mask
    w,h = x.shape
    tmp = x.reshape((w*h, 1), order = "F")
    tmp = np.where(tmp == 1)[0]
    
    start_point = np.concatenate(([0], np.where(np.diff(tmp) != 1)[0]+1))
    end_point = np.where(np.diff(tmp) != 1)[0]
    
    i0 = tmp[start_point] + 1 # add one because the pixel 1 is 1 rather than 0
    i1 = np.concatenate((tmp[end_point], [tmp[-1]])) + 1 # add one because the pixel 1 is 1 rather than 0
    i2 = i1 - i0 + 1 # include starter itself
    
    output_string = ''
    for starter, length in zip(i0, i2):
        output_string = output_string + str(starter) + ' ' + str(length) + ' '
    return output_string.rstrip() # remove the final white space with right strip
    