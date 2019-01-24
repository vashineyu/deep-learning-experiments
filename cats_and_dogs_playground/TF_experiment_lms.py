from __future__ import print_function
# Editor: Seanyu
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import subprocess
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import time
from time import sleep

from threading import Thread, Event, Timer
import queue

from PIL import Image
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0)
parser.add_argument('--image_dir', default="/data/seanyu/cat_dog/dataset/")
parser.add_argument('--save_dir', default='./result')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--do_augment', default=False, type = bool)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--image_size', default= 256, type = int)
parser.add_argument('--n_classes', default=2, type = int)
parser.add_argument('--n_batch', default=100, type = int)
parser.add_argument('--train_ratio', default=0.99, type = float)
parser.add_argument('--use_model_ckpt', default = None, type = str)
parser.add_argument('--model_file_name', default = 'tmp_nb')
parser.add_argument('--n_threads', default = 4, type = int)
parser.add_argument('--dq_size', default = 6, type = int)
parser.add_argument('--use_lms', default = 0, type = int)

FLAGS = parser.parse_args()
FLAGS.image_size = (FLAGS.image_size, FLAGS.image_size, 3)
print(FLAGS)


os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet
if FLAGS.use_lms:
    from tensorflow.contrib.lms import LMS

if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

model_dir = FLAGS.save_dir + '/model'
    
graphs_dir = FLAGS.save_dir + '/graphs'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

"""  Get data """
d_train = FLAGS.image_dir + '/train/'

image_train_list = glob.glob(d_train + '*.jpg')

df_train = pd.DataFrame({'img_path': image_train_list})
df_train['cate'] = df_train.img_path.apply(os.path.basename)
df_train['cate'] = [i.split(".")[0] for i in list(df_train.cate)]
df_train.cate = df_train.cate.replace({'dog': 0, 'cat': 1})

nb_epoch = FLAGS.epochs

df_train_0, df_val_0 = train_test_split(df_train[df_train['cate'] == 0], test_size = 1-FLAGS.train_ratio)
df_train_1, df_val_1 = train_test_split(df_train[df_train['cate'] == 1], test_size = 1-FLAGS.train_ratio)

df_val = pd.concat((df_val_0, df_val_1)).reset_index(drop = True)

del df_val_0, df_val_1

def enqueue(queue, stop, gen_func):
    gen = gen_func()
    while True:
        if stop.is_set():
            return
        queue.put(next(gen))

class create_data_generator():
    def __init__(self,
                 df,
                 open_image_handler,
                 image_size,
                 data_frame_handler = None,
                 nd_inputs_preprocessing_handler = None,
                 batch_size = 32,
                 n_batch = 150,
                 n_classes = 2,
                 dual_ = False, # use epoch generator or not (defualt = False)
                 do_augment = False,
                 aug_params = None):
        
        self.f_readImg = open_image_handler  # how to open image
        self.f_dataproc = data_frame_handler # how to proc original data
        self.f_inputs_preproc = nd_inputs_preprocessing_handler # how to do image preprocessing
        self.bz = batch_size
        self.image_size = image_size
        self.n_batch = n_batch
        self.n_classes = n_classes
        self.aug = aug_params
        self.do_augment = do_augment
        self.img_per_epoch = batch_size * n_batch # make a epoch batch
        self.dual_ = dual_
        
        # run functions at the begin
        # self.df should become list of dataframe anyway
        # if not, do the data_preproc. if yes, pass it
        if data_frame_handler:
            self.df = self.f_dataproc(df)
        else:
            self.df = df
       
    def get_train_data(self):
        while True:
            idxs = self.train_idx_queue.get()
            select_list = []

            for df, idx in zip(self.df, idxs):
                select_list.append(df.iloc[idx])
            select_list = pd.concat(select_list)
            
            x_ = np.array([self.f_readImg(iid, image_size = self.image_size, do_augment = self.do_augment, seq = self.aug) for iid in select_list.img_path], dtype=np.float32)
            x_ = x_.astype(np.float32)
            """ do preprocessing here"""
            if self.f_inputs_preproc:
                x_ = self.f_inputs_preproc(x_)
            else:
                pass

            """ Y out """
            y_ = np.array(select_list['cate'])
            y_ = tf.keras.utils.to_categorical(y_, self.n_classes)

            yield [x_], [y_]
            
    def get_train_epoch(self):
        while True:
            pre_x, pre_y = [], []
            for cumulative_epoch in range(self.n_batch):
                idxs = self.train_idx_queue.get()

                select_list = []

                for df, idx in zip(self.df, idxs):
                    select_list.append(df.iloc[idx])
                select_list = pd.concat(select_list)

                x_ = np.array([self.f_readImg(iid, image_size = self.image_size, do_augment = self.do_augment, seq = self.aug) for iid in select_list.img_path], dtype=np.float32)
                x_ = x_.astype(np.float32)
                """ do preprocessing here"""
                if self.f_inputs_preproc:
                    x_ = self.f_inputs_preproc(x_)
                else:
                    pass

                """ Y out """
                y_ = np.array(select_list['cate'])
                y_ = tf.keras.utils.to_categorical(y_, self.n_classes)
                
                pre_x.append(x_)
                pre_y.append(y_)
                
            yield pre_x, pre_y

    def get_data(self):
        self.mode = 'slow_mode'
        while True:
            if self.dual_:
                # mode check
                if self.train_epoch_queue.qsize() >= 4:
                    self.mode = 'quick_mode'
                    print("In quick mode")
                elif self.train_epoch_queue.qsize() <= 2:
                    if self.mode == 'quick_mode':
                        print('In slow mode')
                    self.mode = 'slow_mode'
            
            if self.mode == "quick_mode":
                # in quick mode
                print("Get data from quick mode, qsize: %i" % self.train_epoch_queue.qsize())
                data = self.train_epoch_queue.get_nowait()
            else:
                # in slow mode
                data = self.train_sample_queue.get()
            
            for ix in np.arange(len(data[0])):
                yield data[0][ix], data[1][ix]
    
    def get_evaluate_data(self, target_df):
        
        x_ = np.array([self.f_readImg(i, image_size = self.image_size, is_training = False) for i in target_df.img_path], dtype=np.float32) # don't do augmentation!
    
        """ do preprocessing here"""
        if self.f_inputs_preproc:
            x_ = self.f_inputs_preproc(x_)
        else:
            pass
        
        """ Y out """
        y_ = np.array(target_df['cate'])
        y_ = tf.keras.utils.to_categorical(y_, num_classes=self.n_classes)
        
        return x_, y_
    
    def _get_train_idx(self):
        """ Description 
        Get training data index for each data frame in the data list
        # note1: self.df should be list of data frame with different categories
        # note2: if there is only 1 class (or for regression problem, should still be embraced [this_df] )
        """
        len_list = [len(df) for df in self.df]
        
        bz_t = self.bz//len(len_list)
        batch_num = [x//bz_t for x in len_list]

        batch_nth = [0] * len(len_list)

        select = [list(range(x)) for x in len_list]

        for s in select:
            random.shuffle(s)

        while True:
            idxs = []
            for i in range(len(len_list)):
                if batch_nth[i] >= batch_num[i]:
                    batch_nth[i] = 0
                    random.shuffle(select[i])
                idx = select[i][batch_nth[i]*bz_t:(batch_nth[i]+1)*bz_t]
                batch_nth[i] += 1
                idxs.append(idx)

            yield idxs
    
    def start_train_threads(self, jobs = 1, dq_size = 10):
        
        if self.dual_:
            self.train_epoch_queue = queue.Queue(maxsize = dq_size)
        self.train_sample_queue = queue.Queue(maxsize = dq_size * 5)
        self.train_idx_queue =queue.Queue(maxsize = dq_size * 100)
        self.jobs = jobs
        ### for stop threads after training ###
        self.events= list()
        self.threading = list()

        ### enqueue train index ###
        event = Event()
        thread = Thread(target = enqueue, 
                        args = (self.train_idx_queue, 
                                event, 
                                self._get_train_idx))
        thread.daemon = True 
        thread.start()
        self.events.append(event)
        self.threading.append(thread)

        ### enqueue train batch ###
        if self.dual_:
            for i in range(jobs):
                event = Event()
                thread = Thread(target = enqueue, args = (self.train_epoch_queue, 
                                                          event, 
                                                          self.get_train_epoch))
                thread.daemon = True 
                thread.start()
                self.events.append(event)
                self.threading.append(thread)
        ### enqueue train samples
        for i in range(jobs // 2):
            event = Event()
            thread = Thread(target = enqueue, args = (self.train_sample_queue,
                                                      event,
                                                      self.get_train_data))
            thread.daemon = True 
            thread.start()
            self.events.append(event)
            self.threading.append(thread)

    def stop_train_threads(self):
        """
        Stop the threading
        """
        # block until all tasks are done
        for t in self.events:
            t.set()
        
        if self.dual_:
            #self.train_epoch_queue.set()
            self.train_epoch_queue.queue.clear()
        
        #self.train_sample_queue.set()
        self.train_sample_queue.queue.clear()
        #self.train_idx_queue.set()
        self.train_idx_queue.queue.clear()
        
        for i, t in enumerate(self.threading):
            t.join(timeout=1)
            print("Stopping Thread %i" % i)

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
            print("== Validation loss has an improvement, save model ==")
            self.min_loss = val_loss
            save_path = saver.save(sess, self.model_name + '.ckpt')
            print("Model saved in path: %s" % save_path)
            
        if not self.save_best_only:
            saver.save(sess, self.model_name + '_' + str(nth_epoch) + '.ckpt',
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


def img_load_and_resize(x, image_size, is_training = True, do_augment = False, seq = None):
    im_w, im_h, im_c = image_size
    im = Image.open(x)
    im = np.array(im.resize((im_w, im_h)))
    if do_augment and is_training:
        im = seq.augment_image(im)
    return im


data_gen = create_data_generator(df=[df_train_0, df_train_1], 
                                 n_classes = FLAGS.n_classes,
                                 image_size = FLAGS.image_size,
                                 do_augment = FLAGS.do_augment,
                                 aug_params=None,
                                 batch_size=FLAGS.batch_size,
                                 n_batch = FLAGS.n_batch,
                                 open_image_handler=img_load_and_resize)

data_gen.start_train_threads(FLAGS.n_threads, dq_size = FLAGS.dq_size)
train_gen = data_gen.get_data()

x_val, y_val = data_gen.get_evaluate_data(df_val)
print(x_val.shape)

def create_model(FLAGS):
    # create a transfer learning model
    tf.reset_default_graph()
    im_w, im_h, im_c = FLAGS.image_size
    
    # placeholders
    drp_holder = tf.placeholder(tf.float32)
    input1 = tf.placeholder(dtype=tf.float32, shape = (None, im_w, im_h, im_c), name = 'input1')
    y_true1 = tf.placeholder(dtype=tf.int8, shape = (None, FLAGS.n_classes), name='y_true1')
    is_training = tf.placeholder(dtype=tf.bool, shape=[])
    lr = tf.placeholder(tf.float32, shape = [])
    is_training = tf.placeholder(tf.bool, shape = [])
    
    # model structs
    with slim.arg_scope(slimNet.resnet_utils.resnet_arg_scope(batch_norm_decay=0.99)):
        _, layers_dict = slimNet.resnet_v2.resnet_v2_50(input1, global_pool=False, 
                                                        is_training=is_training)
        conv_output = layers_dict['resnet_v2_50/block4']
    
    with tf.variable_scope('output'):
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        pred = tf.layers.dense(inputs=x, units=FLAGS.n_classes)
    
    crossentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true1, 
                                                           logits=pred)
    global_loss = tf.reduce_mean(crossentropy)
    
    with tf.name_scope('scope_optimizer'):
        optimizer =  tf.train.AdamOptimizer(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                update = optimizer.minimize(global_loss)

    if FLAGS.use_lms:
        lms_obj = LMS({'scope_optimizer'})
        lms_obj.run(graph = tf.get_default_graph())

    # other
    var_list = tf.trainable_variables()
    all_vars = tf.global_variables() #tf.all_variables() # seems it will depricate after certain version of tensorflow
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    with tf.variable_scope("metrics"):
        pred_output1 = tf.nn.softmax(pred)
        correct_prediction = tf.equal(tf.argmax(pred_output1, 1), 
                                      tf.argmax(y_true1, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # return model as a dictionary, make it easy to access when training or evaluation
    model_key = {'input': [input1],
                 'ground_truth': [y_true1],
                 'output': {'prediction1':pred_output1},
                 'metrics': {'accuracy': accuracy_op}, # add other metrics here (for example, f1, auc)
                 'loss': [global_loss],
                 'update': update,
                 'learning_rate': lr,
                 'is_training': is_training,
                 'intializer': init,
                 'saver': saver, # keep None if no saver
                 'vars': {'partial_vars': var_list, # partial parameters for other usage (for instance, restore)
                          'all_vars': all_vars},
                 'optional': {'dropout': drp_holder}
                 }
    
    metric_history = {k: {'train':[], 'valid':[]} for k in list(model_key['metrics'].keys())}
    
    return model_key, metric_history

model_ops, metric_history = create_model(FLAGS)

class TF_HandyTrainer():
    def __init__(self, model_ops, 
                 data_gen,
                 data_gen_get_data,
                 FLAGS, 
                 metric_history = None, 
                 callback_manager = None,
                 sess = None):
        """ Description
        - model_ops: model graph and its operation key, should be a dict from create_model
        - data_gen: data generator
        - FLAGS: hyper-parameters setting
        - callback_mgr: callback manager, should be a dict
        - callback_handler: give a handler that able to operate when training
        - sess: usually we don't take sess from outside, we init it inside this class
        """
        self.model_ops = model_ops
        self.metric_history = metric_history
        self.data_gen = data_gen
        self.train_gen = data_gen_get_data
        self.FLAGS = FLAGS
        self.callback_mgr = callback_manager
        self.sess = sess
        self.loss_history = {'train': [],
                             'valid': []}
        ### Define and Set train / evaluation ops to list at once
        # Increase certrain ops here
        train_ops = [model_ops['update'], model_ops['loss'][0]]
        valid_ops = [model_ops['loss'][0]]
        if model_ops['metrics'] is not None:
            # append ops if not none
            for i in model_ops['metrics'].keys():
                train_ops.append(model_ops['metrics'][i])
                valid_ops.append(model_ops['metrics'][i])
        # set
        self.train_ops = train_ops
        self.valid_ops = valid_ops

    def initalize(self, graph_dir = None):
        if self.model_ops['saver'] is not None:
            # detect saver
            self.saver = self.model_ops['saver']
        else:
            # no saver, create one
            self.saver = tf.train.Saver()
        
        if self.sess is None:
            self.sess = tf.Session()
        else:
            print("Warning! Use outside session, only do this unless you know it")
            
        print("== INITIALIZE PARAMETERS ==")
        self.sess.run([tf.global_variables_initializer()])
        if graph_dir is not None:
            print("Save graph to " + graph_dir)
            tf.summary.FileWriter(graph_dir, self.sess.graph)
        
    def restore(self, model_to_restore, partial_restore = False):
        """
        Restore weights of layers
        - model_to_restore: should include full path of ckpt
        e.g. tf_pretrain_model/resnet_v2_50.ckpt
        
        """
        print(" ============== ")
        if partial_restore:
            # used in take in pre-trained model
            print("restore paratial parameters")
            # get list of layers to restore and set it into saver
            saver_restore = tf.train.Saver([v for v in self.model_ops['vars']['partial_vars'] if 'resnet_v2_50' in v.name])
            # restore it
            saver_restore.restore(self.sess, model_to_restore)
        else:
            # used in inference
            print("restore all parameters")
            self.saver.restore(self.sess, model_to_restore)
        
    def _train_on_epoch(self, cb_dict):
        # Set learning rate of this epoch
        if 'reduce_lr' in cb_dict.keys():
            epoch_lr = cb_dict['reduce_lr'].lr
        else:
            epoch_lr = self.FLAGS.lr
            
        batch_bar = range(self.FLAGS.n_batch)
        epoch_loss = []
        
        if self.metric_history is not None:
            epoch_metric = {k: [] for k in list(self.metric_history.keys())}
        
        for i in batch_bar:
            #x_, y_ = self.data_gen.train_queue.get()
            x_, y_ = next(self.train_gen)
            batch_result = self.sess.run(self.train_ops, 
                              feed_dict = {self.model_ops['input'][0]: x_,
                                           self.model_ops['ground_truth'][0]: y_,
                                           self.model_ops['is_training']: True,
                                           self.model_ops['learning_rate']: epoch_lr,
                                           self.model_ops['optional']['dropout']: 0.2})
            batch_loss = batch_result[1]
            batch_acc = batch_result[2]
            
            epoch_loss.append(batch_loss)
            current_loss = np.mean(epoch_loss)
            epoch_metric['accuracy'].append(batch_acc)
            
            ### Customized metric calculate over batches ###
            current_acc = np.mean(epoch_metric['accuracy'])
            
            ### Display
            print('\rBatch: %i, Training loss/acc: %.2f/%.2f' % (int(i+1), current_loss, current_acc), end = '')
            
        # return values
        self.metric_history['accuracy']['train'].append(current_acc)
        self.loss_history['train'].append(current_loss)

    
    def evaluate(self, x_, y_ = None):
        """ Description 
        - x_: data to predict
        - y_: data ground truth. if keep None, it is test mode
        """
        bz = self.FLAGS.batch_size
        total_len = range(len(x_) // bz + 1)
        epoch_loss, epoch_predict = [], []
        
        if self.metric_history is not None:
            epoch_metric = {k: [] for k in list(self.metric_history.keys())}
        
        for i in total_len:
            # this is validation mode
            batch_result = self.sess.run(self.valid_ops,
                                      feed_dict = {self.model_ops['input'][0]: x_[i*bz : (i+1) * bz],
                                                   self.model_ops['ground_truth'][0]: y_[i*bz : (i+1) * bz],
                                                   self.model_ops['is_training']: False,
                                                   self.model_ops['optional']['dropout']: 0.0} )
            batch_loss = batch_result[0]
            batch_acc = batch_result[1]

            epoch_metric['accuracy'].append(batch_acc)
            epoch_loss.append(batch_loss)

            current_loss = np.mean([np.mean(i) for i in epoch_loss])
            current_acc = np.mean([np.mean(i) for i in epoch_metric['accuracy']])
        # End of for loop
        # return values

        # validation mode
        self.loss_history['valid'].append(current_loss)
        self.metric_history['accuracy']['valid'].append(current_acc)
        return current_loss, current_acc
        
            
    def predict(self, x_, model_to_restore = None, bz = None):
        """
        Make prediction
        - x_: Input images (np.array) (All images should be pre-processed)
        
        """
        if bz is None:
            # Let it possible to change batch size when make prediction
            bz = self.FLAGS.batch_size
            
        total_len = range(len(x_) // bz + 1)
        epoch_predict = []
        
        self.saver.restore(self.sess, model_to_restore)
        assert model_to_restore is not None, "please pass the model file name (with full path)"
        
        for i in total_len:
            # this is testing mode
            print('testing mode, progress: %i / %i' % (i, np.max(total_len) ))
            batch_predict = self.sess.run([self.model_ops['output']['prediction1']], # prediction
                                      feed_dict = {self.model_ops['input'][0]: x_[i*bz : (i+1) * bz],
                                                   self.model_ops['is_training']: False,
                                                   self.model_ops['optional']['dropout']: 0.0} )
            epoch_predict.append(batch_predict)
        # Reutrn it
        return epoch_predict
                
    
    def do_training(self, validation_set, cb_dict):
        """ Description
        - validation_set: should be a tuple (x, y)
        - cb_dict: callbacks dictionary
        """
        
        #epoch_bar = tqdm(range(self.FLAGS.epochs), desc="epoch", unit="epoch")
        epoch_bar = range(self.FLAGS.epochs)
        for epoch in epoch_bar:
            # train
            #epo_start = time.time()
            _ = self._train_on_epoch(cb_dict = cb_dict) # temporally define as 150
            #epo_end = time.time()
            #epo_ela_time = epo_end - epo_start
            # validation
            _ = self.evaluate(x_ = validation_set[0],
                              y_ = validation_set[1])
            
            # single line report
            print('Epoch: {}/{}'.format(int(epoch+1), self.FLAGS.epochs))
            print('train loss: {} | val loss: {}'.format(self.loss_history['train'][-1], 
                                                        self.loss_history['valid'][-1]))
            
            # run callbacks
            self.callback_mgr.run_on_epoch_end(val_loss = self.loss_history['valid'][-1],
                                               sess = self.sess,
                                               saver = self.saver,
                                               nth_epoch = epoch)
            if 'earlystop' in cb_dict.keys():
                # check there is a earlystop key
                if cb_dict['earlystop'].stop:
                    print("Earlystop criteria met")
                    # met earlystop criteria
                    self.data_gen.stop_train_threads()
                    
                    break
        # Normal stop without earlystop
        self.data_gen.stop_train_threads()


cb_dict = {
    'reduce_lr' : ReduceLROnPlateau(lr=FLAGS.lr, factor=0.5, patience=3),
    'earlystop' : EarlyStopping(min_delta = 1e-4, patience= 15),
    'checkpoint' : Model_checkpoint(model_name=model_dir + '/' +  FLAGS.model_file_name, 
                                    save_best_only=True),
}

callback_dict = {
    'on_session_begin':[], # start of a session
    'on_batch_begin':[], # start of a training batch
    'on_batch_end':[], # end of a training batch
    'on_epoch_begin':[], # start of a epoch
    'on_epoch_end':[cb_dict['earlystop'], 
                    cb_dict['reduce_lr'],
                    cb_dict['checkpoint']], # end of a epoch
    'on_session_end':[] # end of a session
    }
callback_manager = Run_collected_functions(callback_dict)

trainer = TF_HandyTrainer(FLAGS=FLAGS, # hyper-parameters
                          data_gen=data_gen, # data generator
                          data_gen_get_data = train_gen,
                          model_ops=model_ops, # model
                          metric_history=metric_history, # metric recording
                          callback_manager=callback_manager# runable callbacks
                         )

trainer.initalize()
trainer.do_training(cb_dict=cb_dict, validation_set=(x_val, y_val))