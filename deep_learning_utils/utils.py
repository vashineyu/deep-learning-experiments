## python utils

import cv2
import os
import glob
import pandas as pd
import numpy as np

from PIL import Image
import scipy.stats as st
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
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
    
"""
Apply 2d-difference-of-gaussian on mask image
these function can help us easily to get border of each instance from instance mask
"""
def instance_mask_to_binary_mask(img):
    # input: instance mask
    # output: one-hot binary mask
    num_instance = len(np.unique(img)) - 1 # we don't need to take 0 into account
    # allocate memory
    #mask = np.zeros((img.shape[0], img.shape[1], num_instance))
    mask = np.array([img == i for i in np.arange(1, num_instance + 1)])
    mask = np.swapaxes(mask, 0, 1).swapaxes(1, 2)
    mask = mask * 1. # make bool mask into 1/0 mask
    
    return mask

def gkern(k1=11, s1=7, k2 = 11, s2 = 3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*s1+1.)/(k1)
    x = np.linspace(-s1-interval/2., s1+interval/2., k1+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    
    kernel1 = kernel_raw/kernel_raw.sum()
    
    interval = (2*s2+1.)/(k2)
    x = np.linspace(-s2-interval/2., s2+interval/2., k2+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel2 = kernel_raw/kernel_raw.sum()
    
    return kernel1 - kernel2

def apply_2d_dog(onehot_bimask, dog_kernel):
    # onehot-bimask: one instance mask per slice (h,w,instance_mask)
    
    # do filtering
    boundary = np.array([convolve(onehot_bimask[:, :, i], dog_kernel) for i in np.arange(onehot_bimask.shape[-1])])
    boundary = np.swapaxes(boundary, 0, 1).swapaxes(1, 2)
    
    # cut boundary
    boundary[boundary > 0 ] = 0
    boundary[boundary < 0 ] = 1
    
    return boundary.sum(axis = -1) # reduce_dimension

# Example usage
#onehot_mask = instance_mask_to_binary_mask(instance_mask)
#dog_kernel = gkern(k1 = 9, s1 = 7, k2 = 9, s2 = 3)
#tmp = apply_2d_dog(onehot_bimask=a, dog_kernel=dog_kernel)
#tmp[tmp >= 1] = 1 # because overlapping area will larger than one, if we're going to binarized it, just make them as ssame

