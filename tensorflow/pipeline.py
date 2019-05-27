import numpy as np
import tensorflow as tf

x_ph = tf.placeholder(shape=(None, 2), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(x_ph)

iterator = dataset.make_initializable_iterator()

get_next = iterator.get_next()

x = [[1,2],[3,4]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed = {x_ph: x}
sess.run(iterator.initializer, feed)

print(sess.run(get_next))
print(sess.run(get_next))

