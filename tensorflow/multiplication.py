import tensorflow as tf
import numpy as np

x1_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
x2_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
product = tf.multiply(x1_ph, x2_ph)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x1 = np.arange(10)
x2 = np.arange(10)

feed = {x1_ph: x1, x2_ph: x2}
print(sess.run(product, feed))

