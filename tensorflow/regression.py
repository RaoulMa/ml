import numpy as np
import tensorflow as tf
from math import sin
import sys
import matplotlib.pyplot as plt

x_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
y_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

W_init = np.array([0.1]).astype(np.float32)
b_init = np.array([0.1]).astype(np.float32)

W = tf.get_variable(initializer=W_init, name='W')
b = tf.get_variable(initializer=b_init, name='b')

y = tf.add(tf.multiply(W, x_ph), b)

loss = tf.reduce_mean((y - y_ph)**2)

trainable_variables = [W, b]

grads = tf.gradients(ys=loss, xs=trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

train = optimizer.apply_gradients(zip(grads, trainable_variables))

x = np.arange(0,10,0.1).astype(np.float32)
func = lambda x: 2*x + 1.5
noise = np.random.rand(len(x))
t = np.array(list(map(func, x))) + noise

sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed = {x_ph: x, y_ph: t}

for i in range(10000):
    loss_, W_, b_, grads_, _ = sess.run([loss, W, b, grads, train], feed)
    if i % 1000 == 0:
        print('step {} loss {:.2f} W {:.2f} b {:.2f}'.format(i, loss_, W_[0], b_[0]))

y_prediction = sess.run(y, feed)

plt.plot(x, t, linestyle='dotted')
plt.plot(x, y_prediction)
plt.show()

sess.close()

