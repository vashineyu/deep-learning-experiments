import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

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
            
        batch_bar = tqdm(range(self.FLAGS.n_batch), 
                         desc = "Training batch", 
                         unit = "batch", leave = True)
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
            batch_bar.set_description('Batch: %i, Training loss/acc: %.2f/%.2f' % (int(i+1), current_loss, current_acc))
            
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
