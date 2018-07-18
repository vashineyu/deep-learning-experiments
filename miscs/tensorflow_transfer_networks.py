# Restore pnasnet
tf.reset_default_graph()
im_w, im_h, im_c = FLAGS.image_size
input1 = tf.placeholder(dtype=tf.float32, shape = (None, 128, 128, 3), name = 'input1')
y_true1 = tf.placeholder(dtype=tf.float32, shape = (None, 2), name='y_true1')
is_training = tf.placeholder(dtype=tf.bool, shape=[])
lr = tf.placeholder(tf.float32, shape = [])

with slim.arg_scope(pnasnet_large_arg_scope(batch_norm_decay=0.95)):
    _, layers_dict = build_pnasnet_large(images=input1, num_classes=2)
    
exclude = ['aux_7/aux_logits', 'final_layer', 'cell_stem_0/comb_iter_0/left/global_step']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
with tf.Session() as sess:
    saver_restore = tf.train.Saver(variables_to_restore)
    saver_restore.restore(sess, '/data/seanyu/tf-pretrain/pnasnet_5/model.ckpt')
    
   
# Restore nasnet
tf.reset_default_graph()
im_w, im_h, im_c = FLAGS.image_size
input1 = tf.placeholder(dtype=tf.float32, shape = (None, 128, 128, 3), name = 'input1')
y_true1 = tf.placeholder(dtype=tf.float32, shape = (None, 2), name='y_true1')
is_training = tf.placeholder(dtype=tf.bool, shape=[])
lr = tf.placeholder(tf.float32, shape = [])

with slim.arg_scope(nasnet.nasnet_large_arg_scope(batch_norm_decay=0.95)):
    _, layers_dict = nasnet.build_nasnet_large(images=input1, num_classes=2)

exclude = ['aux_11/aux_logits', 'final_layer', 'cell_stem_0/comb_iter_0/left/global_step']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
with tf.Session() as sess:
    saver_restore = tf.train.Saver(variables_to_restore)
    saver_restore.restore(sess, '/data/seanyu/tf-pretrain/nasnet/model.ckpt')
    
    
# Restore resnet_inception
tf.reset_default_graph()
im_w, im_h, im_c = FLAGS.image_size
input1 = tf.placeholder(dtype=tf.float32, shape = (None, 128, 128, 3), name = 'input1') # for resnet_inceptionv2, image must larger than 128, or negative dimension will show
y_true1 = tf.placeholder(dtype=tf.float32, shape = (None, 2), name='y_true1')
is_training = tf.placeholder(dtype=tf.bool, shape=[])
lr = tf.placeholder(tf.float32, shape = [])

#_, layers_dict = nasnet.build_nasnet_large(images=input1, num_classes=2)
with slim.arg_scope(inception_resnet_v2_arg_scope(batch_norm_decay=0.95)):
    _, layers_dict = inception_resnet_v2(inputs=input1, num_classes=2)

exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
with tf.Session() as sess:
    saver_restore = tf.train.Saver(variables_to_restore)
    saver_restore.restore(sess, '/data/seanyu/tf-pretrain/inception_resnet_v2/model.ckpt')
