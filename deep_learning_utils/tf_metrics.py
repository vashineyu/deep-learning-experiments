import tensorflow as tf

def f1sc_metrics(y_true, y_pred):
    prediction = tf.argmax(y_pred, axis = 1)
    truth = tf.argmax(y_true, axis = 1)
    
    TP = tf.count_nonzero(prediction * truth, dtype=tf.float32)
    TN = tf.count_nonzero((prediction - 1) * (truth - 1), dtype=tf.float32)
    FP = tf.count_nonzero(prediction * (truth - 1), dtype=tf.float32)
    FN = tf.count_nonzero((prediction - 1) * truth, dtype=tf.float32)
    
    recall = TP / (TP + FN)
    precision = TP / (TP + TP)
    return 2 * precision * recall / (precision + recall + 1e-8)
