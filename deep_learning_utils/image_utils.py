import cv2
import numpy as np


def cv_load_and_resize(
    x,
    image_size,
    is_training=True,
    seq=None
):
    image_width, image_height, *_ = image_size
    image = cv2.imread(x)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape != image_size:
        image = cv2.resize(image, (image_width, image_height))
    if is_training and seq:
        image = seq.augment_image(image)
    return image

def cv_load_and_pad(
    x,
    image_size,
    is_training=True,
    seq=None
):
    image_width, image_height, *_ = image_size
    image = cv2.imread(x)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape != image_size:
        image = image_crop_and_pad(image, image_height)  # crop and pad to certain size
    if is_training and seq:
        image = seq.augment_image(image)
    return image

def image_crop_and_pad(image, target_size):
    image_width, image_height, *_ = image.shape
    lr_pad = (target_size - (image_width - 1)) // 2  # minus 1 additionally, prevent odd value that pad not enough
    tb_pad = (target_size - (image_height - 1)) // 2

    # Check estimated padding larger than 0
    lr_pad = 0 if lr_pad < 0 else lr_pad
    tb_pad = 0 if tb_pad < 0 else tb_pad
    image = cv2.copyMakeBorder(image, tb_pad, tb_pad, lr_pad, lr_pad, cv2.BORDER_REFLECT)  # WRAP, REPLICATE, REFLECT

    # random crop
    new_height, new_width, *_ = image.shape
    h_rand = 0 if new_height == target_size else np.random.randint(new_height - target_size)
    w_rand = 0 if new_width == target_size else np.random.randint(new_width - target_size)

    image = image[h_rand:(h_rand + target_size), w_rand:(w_rand + target_size), ...]
    return image
