import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet

"""
Description
define your model here, can directly add new function of class of yourself.
However, the output part should follow the model_key and metric_history format
"""

def create_model(FLAGS):
    # create a transfer learning model
    tf.reset_default_graph()
    im_w, im_h, im_c = FLAGS.image_size
    
    # placeholders
    drp_holder = tf.placeholder(tf.float32)
    #input1 = tf.placeholder(dtype=tf.float32, shape = (None, im_w, im_h, im_c), name = 'input1')
    input1 = tf.layers.Input(dtype=tf.float32, shape = (im_w, im_h, im_c), name = 'input1')
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
    
    optimizer =  tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        update = optimizer.minimize(global_loss)
    
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