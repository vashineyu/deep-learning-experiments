import tensorflow as tf
from .backbone import *

graph_mapping = {
    "R-50-v1":ResNet50,
    "R-101-v1":ResNet101,
    "R-152-v1":ResNet152,
    "R-50-v2":ResNet50V2,
    "R-101-v2":ResNet101V2,
    "R-152-v2":ResNet152V2,
    "R-50-xt":ResNeXt50,
    "R-101-xt":ResNeXt101}


def build_model(norm_use, input_shape=(256, 256, 3), num_classes=2, backbone="R-50-v1", weights=None):
    model_fn = graph_mapping[backbone]
    pretrain_modules = model_fn(include_top=False, input_shape=input_shape, norm_use=norm_use, weights=weights)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    logits = tf.keras.layers.Dense(units=num_classes, name="logits")(gap)
    output = tf.keras.layers.Activation("softmax", name="output")(logits)
    
    return tf.keras.Model(inputs=pretrain_modules.input, outputs=output)

def preproc_minmax(img):
    #return (img - img.min()) / (img.max() - img.min())
    return img / 255.

def preproc_resnet(img):
    return tf.keras.applications.resnet50.preprocess_input(img)

def make_optimizer(optimizer, learning_rate):
    """

    Args:
        optimizer (str): optimizer type

    Returns:
        optim: optimizer object

    """
    if optimizer.lower() == "sgd":
        optim = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.95, nesterov=True)
        
    elif optimizer.lower() == "adam":
        optim = tf.keras.optimizers.Adam(lr=learning_rate)
        
    else:
        raise(AssertionError("Optimizer: %s not found" % (optimizer)) )
    
    return optim
