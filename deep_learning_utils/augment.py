import imgaug as ia
from imgaug import augmenters as iaa

# Customize iaa function
def image_channel_suffle(image):
    img_ = image.copy()
    #r, g, b = img_[:,:,0], img_[:,:,1], img_[:,:,2]
    if img_.shape[-1] == 1:
        raise ValueError("Input should be RGB")
    idx = [0,1,2]
    np.random.shuffle(idx)
    #img_[:,:,idx[0]], img_[:,:,idx[1]], img_[:,:,idx[2]] = r, g, b 
    return img_[:,:,idx]

def img_func(images, random_state, parents, hooks):
    for img in images:
        img[:] = image_channel_suffle(img)
    return images

def img_colorswap_func(images, random_state, parents, hooks, ask = None):
    avail_space = {'hsv':cv2.COLOR_RGB2HSV,
                   'hls':cv2.COLOR_RGB2HLS,
                   'lab':cv2.COLOR_RGB2Lab,
                   'luv':cv2.COLOR_RGB2LUV,
                   'xyz':cv2.COLOR_RGB2XYZ,
                   'ycrcb':cv2.COLOR_RGB2YCrCb,
                   'yuv':cv2.COLOR_RGB2YUV}
    for img in images:
        this_swap = avail_space[random.choice(list(avail_space))]
        img[:] = cv2.cvtColor(img, this_swap)
        #plt.imshow(img)
        #plt.show()
    return images

def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images



### Common Usage Setting ###
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5, name="FlipLR"),
    iaa.Flipud(0.5, name="FlipUD"),
    iaa.Multiply((0.8, 1.2), name = "Multiply"),
    iaa.ContrastNormalization((0.8, 1.1), name = "Contrast"),
    iaa.AddToHueAndSaturation((-30,30),name = "Hue"),
    iaa.OneOf([iaa.AdditiveGaussianNoise(scale=0.02*255, name="Noise"),
               iaa.CoarseDropout((0.01, 0.05), size_percent=(0.01, 0.025), name = 'Cdrop')
               ]),
    iaa.Affine(scale = (0.75,1.25), cval=0, mode = 'constant'), # wrap
    iaa.OneOf([iaa.Affine(rotate = 90),
               iaa.Affine(rotate = 180),
               iaa.Affine(rotate = 270)]),
    sometimes(iaa.GaussianBlur((0, 1.5), name="GaussianBlur")),
    iaa.Lambda(img_func, keypoint_func, name = 'channel_swap'),
    sometimes(iaa.PerspectiveTransform(scale=(0.025, 0.100)))
])
