import tensorflow as tf
# Creates a graph.
#with tf.device('/gpu:0'):
    #a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    #b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    #c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
   # print(sess.run(c))

with tf.device('/device:GPU:2'):
    t1 = tf.reshape(t, [-1, ])
    t_return = tf.map_fn(lambda b: tf_interp(b, x, y), t1, dtype=tf.float32, name='t_return')
    t_return = tf.reshape(t_return, [width, height])
    return t_return

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, \
                          log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
