import tensorflow as tf
import numpy as np
import os
import pandas as pd

class EarlyStopping():
    def __init__(self, patience, min_delta = 0.0001):
        # validation loss should at least be less than current min_loss - min_delta
        self.min_delta = min_delta 
        self.patience = patience
        self.epoch_count = 0
        self.min_loss = None
        self.stop = False
        
    def on_epoch_end(self, val_loss, *args, **kwargs):
        if self.min_loss is None or val_loss < self.min_loss - self.min_delta:
            self.min_loss = val_loss
            self.epoch_count = 0
        else:
            self.epoch_count += 1
            
        # if cumulative counts is larger than our patience, set the stop signal to True
        if self.epoch_count >= self.patience:
            self.stop = True
        
class Model_checkpoint():
    def __init__(self, model_name, save_best_only = True):
        self.min_loss = None
        self.model_name = model_name
        self.save_best_only = save_best_only
        
    def on_epoch_end(self, val_loss, nth_epoch, saver, sess, *args, **kwargs):
        if self.min_loss is None or val_loss < self.min_loss:
            self.min_loss = val_loss
            saver.save(sess, 
                       self.model_name + '.ckpt')
        if not self.save_best_only:
            saver.save(sess, 
                       self.model_name + '_' + str(nth_epoch) + '.ckpt',
                       global_step=nth_epoch)
        
class ReduceLROnPlateau():
    def __init__(self, lr, factor, patience, min_lr = 1e-10):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_loss = None
        self.epoch_count = 0
    
    def on_epoch_end(self, val_loss, *args, **kwargs):
        if self.min_loss is None or val_loss < self.min_loss:
            epoch_count = 0
            self.min_loss = val_loss
        else:
            self.epoch_count += 1
        
        if self.epoch_count == self.patience:
            self.lr *= self.factor
            self.epoch_count = 0
            
            if self.lr <= self.min_lr:
                self.lr = self.min_lr
                
class Run_collected_functions():
    def __init__(self, callback_dicts):
        self.on_session_begin = callback_dicts['on_session_begin']
        self.on_session_end = callback_dicts['on_session_end']
        self.on_batch_begin = callback_dicts['on_batch_begin']
        self.on_batch_end = callback_dicts['on_batch_end']
        self.on_epoch_begin = callback_dicts['on_epoch_begin']
        self.on_epoch_end = callback_dicts['on_epoch_end']
        
    def run_on_epoch_end(self, val_loss, nth_epoch = None, sess = None, saver = None):
        for func in self.on_epoch_end:
            getattr(func, 'on_epoch_end')(val_loss = val_loss,
                                          nth_epoch = nth_epoch,
                                          sess = sess,
                                          saver = saver)
        
    def run_on_session_end(self, *args, **kwargs):
        pass