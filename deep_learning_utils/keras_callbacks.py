from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class LogAUC(Callback):
    def __init__(self):
        return

    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        if "val_auc" not in self.params['metrics']:
            self.params['metrics'].append("val_auc")
            
    def on_epoch_end(self, epoch, logs = {}):
        logs = logs or {}
        #y_pred = self.model.predict(self.validation_data[0])
        y_pred = self.model.predict(self.validation_data[0])
        
        #self.aucs.append(auc)
        logs["val_auc"] = roc_auc_score(self.validation_data[1], y_pred)

class f1sc(Callback):
    def __init__(self):
        return

    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        if "val_f1sc" not in self.params['metrics']:
            # f1sc
            self.params['metrics'].append("val_f1sc")
        if "val_fp" not in self.params['metrics']:
            # false-positive (false alarm)
            self.params['metrics'].append("val_fp")
        if "val_fn" not in self.params['metrics']:
            # false-negative (miss)
            self.params['metrics'].append("val_fn")
        if "val_tp" not in self.params['metrics']:
            # true-positive (hit)
            self.params['metrics'].append("val_tp")
        if "val_tn" not in self.params['metrics']:
            # trun-negative (correct reject)
            self.params['metrics'].append("val_tn")
            
    def on_epoch_end(self, epoch, logs = {}):
        logs = logs or {}
        thres = 0.5 # prepare it, for further we can change our judgement threshold
        y_true = self.validation_data[1].argmax(axis = 1)
        y_pred = self.model.predict(self.validation_data[0])
        
        y_pred = (y_pred[:, 1] > thres) * 1
        con_martrix = confusion_matrix(y_true= y_true, y_pred= y_pred)
        
        ### 0 1 (pred)
        # 0
        # 1
        # (true)
        
        tp = con_martrix[1][1]
        tn = con_martrix[0][0]
        fp = con_martrix[0][1]
        fn = con_martrix[1][0]
        
        logs["val_f1sc"] = f1_score(y_true = y_true, y_pred = y_pred)
        logs["val_tp"] = tp
        logs["val_tn"] = tn
        logs["val_fp"] = fp
        logs["val_fn"] = fn
