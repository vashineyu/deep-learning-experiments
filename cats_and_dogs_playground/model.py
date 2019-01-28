import tensorflow as tf
from backbone import *


def build_model(model_fn, norm_use, input_shape=(256,256,3), num_classes=2, weights=None):
    pretrain_modules = model_fn(include_top=False, input_shape=input_shape, norm_use=norm_use, weights=weights)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    logits = tf.keras.layers.Dense(units=num_classes, name="logits")(gap)
    output = tf.keras.layers.Activation("softmax", name="output")(logits)
    
    return tf.keras.Model(inputs=pretrain_modules.input, outputs=output)

"""
class Build_FunctionalModel():
    def __init__(model_fn, norm_use, input_shape=(256,256,3), num_class=2, weights=None):
        
        self.pretrain_modules = model_fn(include_top=False, input_shape=input_shape, norm_use=norm_use, weights=weights)
        gap = tf.keras.layers.GlobalAveragePooling2D()(self.pretrain_modules.output)
        self.logit = tf.keras.layers.Dense(units=num_class, name="logit")(gap)
        self.out = tf.keras.layers.Activation("softmax", name="output")(self.logit)
        
    def build():
        return tf.keras.models.Model(inputs=[self.pretrain_modules.input],
                                     outputs=[self.out])
"""