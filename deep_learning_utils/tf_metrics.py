import tensorflow as tf


def get_accuracy(y_true, y_pred):
    """
    Args:
        y_true should be one-hot encoded tensor
        y_pred should be logit tensor
    Reutrn:
        accuracy
    """
    correct_predict = tf.equal(
        tf.argmax(y_true, axis=1),
        tf.argmax(y_pred, axis=1),
    )
    return tf.reduce_mean(tf.cast(correct_predict, tf.float16))

def get_f1sc(y_true, y_pred, threshold=0.5):
    """
    Args:
        y_true: should be one-hot encoded tensor
        y_pred: should be logit tensor
        threshold: thres to claim it is 1

    Reutrn:
        f1-score under threshold
    """
    y_true = tf.argmax(y_true, axis=1)
    y_pred = y_pred[:, 1] >= threshold

    recall = tf.metrics.recall(y_true, y_pred)[1]
    precision = tf.metrics.precision(y_true, y_pred)[1]

    return 2 * (recall * precision) / (recall + precision)


def get_auc(y_true, y_pred, ctype="ROC"):
    """
    Args:
        y_true: should be single vector
        y_pred: should be single vector as 1
        ctype: ROC or PR (only support ROC currently)
    Reutrn:
        Area Under Curve of ROC/PR
    """
    if len(y_true.shape) == 2:
        y_true = tf.argmax(y_true, axis=1)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]

    auc = tf.metrics.auc(
        labels=y_true,
        predictions=y_pred,
        num_thresholds=500,
        curve=ctype,
    )[1]
    return auc
