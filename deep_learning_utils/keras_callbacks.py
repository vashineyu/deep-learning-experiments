from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


class logAUC(Callback):
    def __init__(self):
        return

    def on_epoch_begin(self, epoch, logs={}):
        logs = logs or {}
        record_items = ['val_auc', 'val_f1sc', 'val_fp', 'val_fn', 'val_tp', 'val_tn']
        for i in record_items:
            if i not in self.params['metrics']:
                self.params['metrics'].append(i)

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        y_true = self.validation_data[1].argmax(axis=1)
        y_pred = self.model.predict(self.validation_data[0])
        logs['val_auc'] = roc_auc_score(y_true, y_pred[:, 1])

        thres = 0.5  # prepare it, for further we can change our judgement threshold
        y_pred = (y_pred[:, 1] >= thres) * 1

        con_martrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

        logs['val_tp'] = con_martrix[1][1]
        logs['val_tn'] = con_martrix[0][0]
        logs['val_fp'] = con_martrix[0][1]
        logs['val_fn'] = con_martrix[1][0]
        logs['val_f1sc'] = f1_score(y_true=y_true, y_pred=y_pred)
