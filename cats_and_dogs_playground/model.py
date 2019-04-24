import sys
import tensorflow as tf
sys.path.append("/mnt/deep-learning/usr/seanyu/lab_mldl_tools")
from models.tf_resnet.model import *

graph_mapping = {
    "R-50-v1":ResNet50,
    "R-101-v1":ResNet101,
    "R-152-v1":ResNet152,
    "R-50-v2":ResNet50V2,
    "R-101-v2":ResNet101V2,
    "R-152-v2":ResNet152V2,
    "R-50-xt":ResNeXt50,
    "R-101-xt":ResNeXt101}


def build_model(norm_use, input_shape=(256,256,3), num_classes=2, backbone="R-50-v1", weights=None):
    model_fn = graph_mapping[backbone]
    pretrain_modules = model_fn(include_top=False, input_shape=input_shape, norm_use=norm_use, weights=weights)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    logits = tf.keras.layers.Dense(units=num_classes, name="logits")(gap)
    output = tf.keras.layers.Activation("softmax", name="output")(logits)
    
    return tf.keras.Model(inputs=pretrain_modules.input, outputs=output)

def preproc(img):
    #return (img - img.min()) / (img.max() - img.min())
    return img / 255.

def make_optimizer(cfg):
    if cfg.MODEL.OPTIMIZER.lower() == "sgd":
        optim = tf.keras.optimizers.SGD(lr=cfg.TRAIN.LR, momentum=0.95, nesterov=True)
        
    elif cfg.MODEL.OPTIMIZER.lower() == "adam":
        optim = tf.keras.optimizers.Adam(lr=cfg.TRAIN.LR)
        
    else:
        raise(AssertionError("Optimizer: %s not found" % (cfg.MODEL.OPTIMIZER)) )
    
    return optim