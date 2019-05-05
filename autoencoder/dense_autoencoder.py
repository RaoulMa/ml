import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data', validation_size=0)

encoding_dim = 32
image_size = mnist.train.images.shape[1]

inputs_ph = tf.placeholder(tf.float32, (None, image_size), name='inputs')
targets_ph = tf.placeholder(tf.float32, (None, image_size), name='targets')

encoded = tf.layers.dense(inputs_ph, encoding_dim, activation=tf.nn.relu)
logits = tf.layers.dense(encoded, image_size, activation=None)
decoded = tf.nn.sigmoid(logits, name='output')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_ph, logits=logits))
train = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_epochs = 1
batch_size = 200
for e in range(n_epochs):
    for b in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        feed = {inputs_ph: batch[0], targets_ph: batch[0]}
        loss_, _ = sess.run([loss, train], feed_dict=feed)
    print("ep {} batch {} loss {:.2f}".format(e+1, b, loss_))

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
test_imgs = mnist.test.images[:10]
reconstructed, compressed = sess.run([decoded, encoded], feed_dict={inputs_ph: test_imgs})

for images, row in zip([test_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()

sess.close()