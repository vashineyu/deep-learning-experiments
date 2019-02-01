import tensorflow as tf
from backbone import *


def build_model(model_fn, norm_use, input_shape=(256,256,3), num_classes=2, weights=None):
    pretrain_modules = model_fn(include_top=False, input_shape=input_shape, norm_use=norm_use, weights=weights)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    logits = tf.keras.layers.Dense(units=num_classes, name="logits")(gap)
    output = tf.keras.layers.Activation("softmax", name="output")(logits)
    
    return tf.keras.Model(inputs=pretrain_modules.input, outputs=output)

def preproc(img):
    #return (img - img.min()) / (img.max() - img.min())
    return img / 255.

def parse_model_fn(model_name):
    if model_name == "R-50-v1":
        model_fn = ResNet50
        
    elif model_name == "R-101-v1":
        model_fn = ResNet101
        
    elif model_name == "R-152-v1":
        model_fn = ResNet152
        
    elif model_name == "R-50-v2":
        model_fn = ResNet50V2
        
    elif model_name == "R-101-v2":
        model_fn = ResNet101V2
    
    elif model_name == "R-152-v2":
        model_fn = ResNet152V2
        
    elif model_name == "R-50-xt":
        model_fn = ResNeXt50
        
    elif model_name == "R-101-xt":
        model_fn = ResNeXt101
    
    else:
        raise(AssertionError("Model: %s not found, check your config" % (model_name)))
        
    return model_fn

def make_optimizer(cfg):
    if cfg.MODEL.OPTIMIZER.lower() == "sgd":
        optim = tf.keras.optimizers.SGD(lr=cfg.TRAIN.LR, momentum=0.95, nesterov=True)
        
    elif cfg.MODEL.OPTIMIZER.lower() == "adam":
        optim = tf.keras.optimizers.Adam(lr=cfg.TRAIN.LR)
        
    else:
        raise(AssertionError("Optimizer: %s not found" % (cfg.MODEL.OPTIMIZER)) )
    
    return optim