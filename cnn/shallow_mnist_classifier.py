#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf;
#from tensorflow.examples.tutorials.mnist import input_data

#set of handwritten images with 28x28 = 284 pixels
mnist = input_data.read_data_sets("data/", one_hot=True)

#mnist.train: [55000,784] array 
#mnist.labels: [55000,10] array 

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#model: y = W*x+b
y = tf.nn.softmax(tf.matmul(x, W) + b)

#cross entropy: sum(-y_*log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#gradient descent with learning rate 0.5 to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#train model
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#test accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
