import subprocess
from threading import Thread, Event
import queue
import pandas as pd
import numpy as np
import cv2
import random
import tensorflow as tf

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
            self.data_gen.train_epoch_queue.queue.clear()
        
        #self.train_sample_queue.set()
        self.train_sample_queue.queue.clear()
        #self.train_idx_queue.set()
        self.train_idx_queue.queue.clear()
        
        for i, t in enumerate(self.threading):
            t.join(timeout=1)
            print("Stopping Thread %i" % i)