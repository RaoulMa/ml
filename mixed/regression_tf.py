#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
y_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

linear_model =  W * x_ph + b

loss = tf.reduce_mean(tf.square(linear_model - y_ph), axis=0)

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

x = [1, 2, 3, 4]
y = [0, -1, -2, -3]
feed = {x_ph: x, y_ph: y}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2000):
  sess.run(train, feed)

W_, b_, loss_ = sess.run([W, b, loss], feed)
print("W: %s b: %s loss: %s"%(W_, b_, loss_))

sess.close()