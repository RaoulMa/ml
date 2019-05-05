#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

x_ph = tf.placeholder(tf.float32, [None, 784])
y_ph = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x_ph, W) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y_ph * tf.log(y), axis=1))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_ph,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_batches = 500
batch_size = 128
log_step = 50
for b in range(n_batches):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    loss_, _ = sess.run([loss, train_step], feed_dict={x_ph: batch_xs, y_ph: batch_ys})
    if b % log_step == 0:
        accuracy_ = sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_ph: mnist.test.labels})
        print('batch {} loss {:.2f} acc {:.2f}'.format(b, loss_, accuracy_))

accuracy_ = sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_ph: mnist.test.labels})
print('final accuracy {}'.format(accuracy_))

sess.close()