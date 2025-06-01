import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
labels = mnist.test.labels[:1000, :]

sess = tf.Session()
# Placeholders
z_dim = 96
y_dim = 10
z = tf.placeholder(tf.float32, shape=[None, z_dim], name='noise')
y = tf.placeholder(tf.float32, shape=[None, y_dim], name='labels')

# Build generator graph
G_sample = generator_conv(z, y, reuse=False)


def activation(x, alpha=0.01):
    """Leaky ReLU activation."""
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    """Generate random noise uniformly between -1 and 1."""
    return tf.random_uniform(shape=[batch_size, dim], minval=-1, maxval=1)


def generator_conv(z, c, reuse=False):
    """
    Generator network: builds a conv transpose generator conditioned on labels c.

    Args:
        z: Noise vector input tensor [batch_size, noise_dim].
        c: Conditioning labels tensor [batch_size, y_dim].
        reuse: Whether to reuse variables in the scope.

    Returns:
        Generated images tensor reshaped to [batch_size, 784].
    """
    with tf.variable_scope("generator_conv", reuse=reuse):
        keep_prob = 0.5
        d1 = 4
        d2 = 256

        x = tf.concat([z, c], axis=1)
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, rate=1 - keep_prob)

        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])

        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=1 - keep_prob)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=1 - keep_prob)

        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=5, strides=1, padding='same', activation=activation)

        x = tf.tanh(x)
        x = tf.reshape(x, shape=[-1, 784])
        return x


def save_generator(sess, path='my_generator_model.ckpt'):
    """Save the generator variables to a checkpoint file."""
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv')
    saver = tf.train.Saver(var_list=g_vars)
    saver.save(sess, path)
    print('Generator model saved to', path)


def load_generator(sess, path='my_generator_model.ckpt'):
    """Restore the generator variables from a checkpoint file."""
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv')
    saver = tf.train.Saver(var_list=g_vars)
    saver.restore(sess, path)
    print('Generator model restored from', path)


def generate_fake_samples(sess, labels, num_samples):
    """
    Generate fake images conditioned on labels.

    Args:
        sess: TensorFlow session.
        labels: numpy array of labels, shape [num_samples, y_dim].
        num_samples: Number of samples to generate.

    Returns:
        Generated samples as numpy array of shape [num_samples, 784].
    """
    load_generator(sess)
    selected_labels = labels[:num_samples]
    noise_batch = sess.run(sample_noise(num_samples, z_dim))
    generated = sess.run(G_sample, feed_dict={z: noise_batch, y: selected_labels})
    return generated


# Example usage:
if __name__ == '__main__':
    sess.run(tf.global_variables_initializer())
    # You should first train your generator and save it using save_generator(sess) before calling generate_fake_samples

    # For demo: generate 10 samples
    fake_samples = generate_fake_samples(sess, labels, 10)
    print('Generated samples shape:', fake_samples.shape)
