#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Deep MNIST classifier with 99% test accuracy. The architecture is given by:
Input: images with 28x28 pixels, 1 color number
1. Layer: Conv1 -> ReLu -> MaxPool: [.,14,14,32] 
2. Layer: Conv2 -> ReLu -> MaxPool: [.,7,7,64]
3. Layer: FC -> ReLu: [.,1024]
4. Layer: FC -> ReLu: [.,10]
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_ph = tf.placeholder(tf.float32, shape=[None, 784])
y_ph = tf.placeholder(tf.float32, shape=[None, 10])

### 1. layer: conv and maxpool ###
# convolution with 5x5 patch, 32 features: stride = 1, paddings = same
# maxpool with 2x2 patch, stride = 2, padding = same
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x_ph, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output [None,28,28,32]
h_pool1 = max_pool_2x2(h_conv1) # output [None,14,14,32] 

### 2. layer: conv and maxpool ###
# convolution with 5x5 patch, 64 features: stride = 1, paddings = same
# maxpool with 2x2 patch, stride = 2, padding = same
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output [None,14,14,64]
h_pool2 = max_pool_2x2(h_conv2) # output [None,7,7,64]

### 3. layer: FC 
# fully connected layer with 1024 nodes
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#implement dropout to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

###4. layer: FC
# fully connected layer with 10 nodes
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

### define loss function and train the model ###
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_ph, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_batches = 1000
batch_size = 64
log_step = 100
for i in range(n_batches):
    batch = mnist.train.next_batch(batch_size)
    if i % log_step == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x_ph: batch[0], y_ph: batch[1], keep_prob: 1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_ph: mnist.test.labels, keep_prob: 1.0})
        print('step %d, train accuracy %g, test accuracy %g' % (i, train_accuracy,test_accuracy))
    sess.run(train_step, feed_dict={x_ph: batch[0], y_ph: batch[1], keep_prob: 0.5})

sess.close()

      

    
    
    
    
