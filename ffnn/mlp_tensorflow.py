"""
Import libraries and perform data preprocessing.
"""
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime
import os
import sklearn.model_selection
import pandas as pd
#%matplotlib inline

# Print current python and tensorflow versions.
print('Tensorflow version', tf.__version__)
print('Python version', sys.version)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert dense labels to one-hot-encodings."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_dense(labels_one_hot):
    """Convert one-hot encodings into dense labels."""
    return np.argmax(labels_one_hot,1)

def get_next_mini_batch():
    """Return the next mini batch of training data.
    
    Returns
        x_trn_mb: mini batch of training x values
        y_trn_mb: batch of training y values
    """
    global idx_in_epoch, epoch, ids, x_trn, y_trn
    
    start = idx_in_epoch
    idx_in_epoch += mb_size           
    epoch += mb_size/len(x_trn)

    # At the start of each epoch.
    if start == 0:
        np.random.shuffle(ids)  

    # In case the current index is larger than one epoch.
    if idx_in_epoch > len(x_trn):
        idx_in_epoch = 0
        epoch -= mb_size/len(x_trn) 
        return get_next_mini_batch() # Recursive use of function.

    # Take mini batch from training samples excluding validation samples
    x_trn_mb = x_trn[ids[start:idx_in_epoch]]
    y_trn_mb = y_trn[ids[start:idx_in_epoch]]
    return x_trn_mb, y_trn_mb

# number of training and test samples
n_train = 1000 # train samples per class
n_test = 1000 # test samples per class
n_classes = 2  
r_seed = 123
n_hid_1 = 10

# create distinct classes
np.random.seed(r_seed)
tf.set_random_seed(r_seed)

# create training set
x_train = np.reshape(np.concatenate([np.random.normal(-1,1,n_train), np.random.normal(1,1,n_train)]),(-1,1))
y_train = np.concatenate([np.array([0]*n_train), np.array([1]*n_train)])

# create test set
x_test = np.reshape(np.concatenate([np.random.normal(-1,1,n_test), np.random.normal(1,1,n_test)]),(-1,1))
y_test = np.concatenate([np.array([0]*n_test), np.array([1]*n_test)])

# randomize training and test data
ids = np.arange(len(x_train))
np.random.shuffle(ids)
x_train = x_train[ids]
y_train = y_train[ids]

ids = np.arange(len(x_test))
np.random.shuffle(ids)
x_test = x_test[ids]
y_test = y_test[ids]

# one-hot-encoding
y_train = dense_to_one_hot(y_train, n_classes)
y_test = dense_to_one_hot(y_test, n_classes)

print('train data shapes: ', x_train.shape, y_train.shape)
print('test data shapes: ', x_test.shape, y_test.shape)

# visualize training data
bins = np.linspace(-4, 4, 50)
plt.hist([[x for i,x in enumerate(x_train[:,0]) if y_train[i,0]==1],
         [x for i,x in enumerate(x_train[:,0]) if y_train[i,1]==1]], 
         bins=bins, alpha=0.5, label=['class 0', 'class 1'])
plt.legend(loc='upper right')
plt.show()
    

"""
Load an existing graph and extend the graph by one additional layer.
"""

learn_rate = 0.001            # learn rate
idx_in_epoch = 0              # current index in epoch
mb_size = 50                  # mini batch size
epoch = 0.                    # current epoch
n_hid2 = 10  

def initialize_uninitialized(sess):
    """Initialize unitialized variables"""
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print('Initialize: ',[str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

# Reset current graph.
tf.reset_default_graph()

# Load existing graph.
filepath = os.path.join('.', 'saves', 'mlp_layer_1')
saver_tf = tf.train.import_meta_graph(filepath + '.meta')

# Load tensors from graph.
graph = tf.get_default_graph() # save default graph
a1_tf = graph.get_tensor_by_name('a1_tf:0')
y_data_tf = graph.get_tensor_by_name('y_data_tf:0')
x_data_tf = graph.get_tensor_by_name('x_data_tf:0')

# Fix a1_tf, i.e. no gradient descent beyond this point.
# Otherwise, the whole graph will be retrained
#a1_tf = tf.stop_gradient(a1_tf)

# Additional second layer.
w2_tf = tf.Variable(tf.truncated_normal([a1_tf.shape[1].value,n_hid2], stddev=0.1), name='w2_tf')
b2_tf = tf.Variable(tf.truncated_normal([n_hid2], stddev=0.1), name='b2_tf')
z2_tf = tf.add(tf.matmul(a1_tf,w2_tf), b2_tf, name='z2_tf')
a2_tf = tf.nn.relu(z2_tf, name='a2_tf')

# Output layer.
w3_tf = tf.Variable(tf.truncated_normal([a2_tf.shape[1].value,n_classes], stddev=0.1), name='w3_tf')
b3_tf = tf.Variable(tf.truncated_normal([n_classes], stddev=0.1), name='b3_tf')
z3_tf = tf.add(tf.matmul(a2_tf,w3_tf), b3_tf, name='z3_tf')

# Softmax result in terms of probablities ("one-hot" encoding)
y_prob_tf = tf.nn.softmax(z3_tf, name='y_prob_tf')

# Dense results in terms of classes (dense encoding)
y_class_tf = tf.argmax(y_prob_tf, 1, name='y_class_tf')

# Loss function
loss_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data_tf, logits=z3_tf), name='loss_tf')

# Optimization.
#optimize_tf = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss_tf, name='optimize_tf')
optimize_tf = tf.train.RMSPropOptimizer(learning_rate=learn_rate).minimize(loss_tf, name='optimize_tf')

# Tensor of correct predictions
correct_tf = tf.equal(y_class_tf, tf.argmax(y_data_tf,1), name='correct_tf')  

# Score: accuracy
score_tf = tf.reduce_mean(tf.cast(correct_tf, dtype=tf.float32), name='score_tf')

# Create new session with loaded graph.
sess = tf.Session() # default session
saver_tf.restore(sess, filepath) # restore session

# Initialize only uninitialized variables
initialize_uninitialized(sess)

# Use 10-fold cross validation.
cv_num = 10
kfold = sklearn.model_selection.KFold(cv_num, shuffle=True, random_state=r_seed)

for i,(train_ids, valid_ids) in enumerate(kfold.split(x_train)):

    # Samples used for training and evaluation
    x_trn = x_train[train_ids]
    y_trn = y_train[train_ids]
    x_vld = x_train[valid_ids]
    y_vld = y_train[valid_ids]

    # Reset global parameters. 
    epoch = 0.
    idx_in_epoch = 0
    ids = np.arange(len(x_trn))

    # Train the MLP classifier
    for n in range(int(20*len(x_trn)/mb_size)):

        x_batch, y_batch = get_next_mini_batch()
        sess.run(optimize_tf, feed_dict={x_data_tf: x_batch, y_data_tf: y_batch})

        if n%int(1*len(x_trn)/mb_size)==0:
            train_loss, train_score = sess.run([loss_tf, score_tf],
                                               feed_dict = {x_data_tf:x_batch, y_data_tf:y_batch})
            valid_loss, valid_score = sess.run([loss_tf, score_tf],
                                               feed_dict = {x_data_tf:x_vld, y_data_tf:y_vld})
            print('{:.3f} epoch: train/valid loss {:.3f}/{:.3f}, train/valid score {:.3f}/{:.3f}'.format(
                epoch, train_loss, valid_loss, train_score, valid_score))
    break

# Predictions of loaded graph.
y_train_pred, train_loss, train_score = sess.run([y_class_tf, loss_tf, score_tf], 
                                                 feed_dict={x_data_tf:x_train, y_data_tf:y_train})
y_test_pred, test_loss, test_score = sess.run([y_class_tf, loss_tf, score_tf], 
                                              feed_dict={x_data_tf:x_test, y_data_tf:y_test})
# Print losses and scores.
print('\nPredictions of loaded graph:\ntrain/test loss {:.3f}/{:.3f}, train/test score {:.3f}/{:.3f}'.format(
    train_loss, test_loss, train_score, test_score))

sess.close()


