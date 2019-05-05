import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data', validation_size=0)

inputs_ph = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ph = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

### Encoder
conv1 = tf.layers.conv2d(inputs_ph, 16, (3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x16
maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
# Now 14x14x16
conv2 = tf.layers.conv2d(maxpool1, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x8
maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
# Now 7x7x8
conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x8
encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
# Now 4x4x8

### Decoder
upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
# Now 7x7x8
conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x8
upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
# Now 14x14x8
conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x8
upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
# Now 28x28x8
conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x16

logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
#Now 28x28x1

decoded = tf.nn.sigmoid(logits, name='decoded')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_ph, logits=logits))
train = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_epochs = 10
batch_size = 128
noise_factor = 0.5
for e in range(n_epochs):
    for b in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1))

        # add noise to images
        imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        imgs = np.clip(imgs, 0., 1.)

        feed = {inputs_ph: imgs, targets_ph: imgs}
        loss_, _ = sess.run([loss, train], feed_dict=feed)
    print("ep {} batch {} loss {:.2f}".format(e+1, b, loss_))

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
test_imgs = mnist.test.images[:10].reshape((-1,28,28,1))
test_imgs = test_imgs + noise_factor * np.random.randn(*test_imgs.shape)
test_imgs = np.clip(test_imgs, 0., 1.)
reconstructed, compressed = sess.run([decoded, encoded], feed_dict={inputs_ph: test_imgs})

for images, row in zip([test_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()

sess.close()