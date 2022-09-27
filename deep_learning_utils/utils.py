import cv2
import numpy as np
import scipy.stats as st
from scipy.ndimage.filters import convolve


def label_reconstruct(x):
    tmp = x.split(' ')
    start_point = tmp[0::2]
    expand_len = tmp[1::2]
    mask = []
    for i, j in zip(start_point, expand_len):
        cur = list(range(int(i), int(i) + int(j)))
        mask.extend(cur)
    return mask

def label_to_mask(x, image_size):
    image_width, image_height = image_size[:2]
    x = [ix - 1 for ix in x]

    # background
    background = np.ones((image_width * image_height))
    background[x] = 0
    background = background.reshape((image_width, image_height), order="F")

    # foreground
    mask = np.zeros((image_width * image_height))
    mask[x] = 1
    mask = mask.reshape((image_width, image_height), order="F")

    images = np.dstack((background, mask))

    return images


def onehot_to_mask(x, input_shape, output_shape, n_classes=2):
    width, height = input_shape[:2]
    target_image_width, target_image_height = output_shape[:2]

    x = x.reshape((width, height, n_classes))
    x = cv2.resize(x, (target_image_height, target_image_width))
    x = x.argmax(axis=-1)
    return x

def mask_to_label(x):
    # the input should be a mask
    height, width = x.shape[:2]
    tmp = x.reshape((height * width, 1), order="F")
    tmp = np.where(tmp == 1)[0]

    start_point = np.concatenate(([0], np.where(np.diff(tmp) != 1)[0] + 1))
    end_point = np.where(np.diff(tmp) != 1)[0]

    i0 = tmp[start_point] + 1  # add one because the pixel 1 is 1 rather than 0
    i1 = np.concatenate((tmp[end_point], [tmp[-1]])) + 1  # add one because the pixel 1 is 1 rather than 0
    i2 = i1 - i0 + 1  # include starter itself

    output_string = ''
    for starter, length in zip(i0, i2):
        output_string = output_string + str(starter) + ' ' + str(length) + ' '
    return output_string.rstrip()  # remove the final white space with right strip


def instance_mask_to_binary_mask(img):
    num_instance = len(np.unique(img)) - 1  # we don't need to take 0 into account

    # allocate memory
    mask = np.array([img == i for i in np.arange(1, num_instance + 1)])
    mask = np.swapaxes(mask, 0, 1).swapaxes(1, 2)
    mask = mask * 1.  # make bool mask into 1/0 mask
    return mask

def gkern(k1=11, s1=7, k2=11, s2=3):
    interval = (2 * s1 + 1.) / k1
    x = np.linspace(-s1 - interval / 2., s1 + interval / 2., k1 + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))

    kernel1 = kernel_raw / kernel_raw.sum()

    interval = (2 * s2 + 1.) / k2
    x = np.linspace(-s2 - interval / 2., s2 + interval / 2., k2 + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel2 = kernel_raw / kernel_raw.sum()

    return kernel1 - kernel2

def apply_2d_dog(onehot_bimask, dog_kernel):
    # do filtering
    boundary = np.array([
        convolve(onehot_bimask[:, :, i], dog_kernel) for i in np.arange(onehot_bimask.shape[-1])
    ])
    boundary = np.swapaxes(boundary, 0, 1).swapaxes(1, 2)

    # cut boundary
    boundary[boundary > 0] = 0
    boundary[boundary < 0] = 1

    return boundary.sum(axis=-1) # reduce_dimension
