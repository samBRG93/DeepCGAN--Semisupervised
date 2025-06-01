import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
import tf_layers  # Assumendo che tf_layers sia il modulo contenente Batchnorm e conv_transpose
from tf_layers import Batchnorm, conv_transpose

# Caricamento dati MNIST
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
labels = mnist.test.labels[:1000, :]

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def define_placeholders(noise_dim=96, label_dim=10):
    """Definisce i placeholder per rumore e etichette."""
    z = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise')
    y = tf.placeholder(tf.float32, shape=[None, label_dim], name='labels')
    return z, y


def sample_noise(batch_size, dim):
    """Campiona rumore uniforme tra -1 e 1."""
    return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)


def generator_conv(z, y, reuse=True):
    """Generatore convoluzionale condizionato con batch normalization e LeakyReLU."""
    batch_size = tf.shape(z)[0]  # usa tf.shape per compatibilità dinamica

    output_height, output_width = 64, 64
    scale_h = np.array(
        [output_height / 16, output_height / 8, output_height / 4, output_height / 2, output_height]).astype(int)
    scale_w = np.array([output_width / 16, output_width / 8, output_width / 4, output_width / 2, output_width]).astype(
        int)

    with tf.variable_scope("generator_conv", reuse=reuse):
        init = tf.contrib.layers.xavier_initializer()

        x = tf.concat([z, y], axis=1)

        # Dense layer iniziale e reshape
        x = tf.layers.dense(x, units=4 * 4 * 128, kernel_initializer=init)
        x = Batchnorm('Generator.BN0', [0], x)
        x = tf.nn.leaky_relu(x)
        x = tf.reshape(x, [-1, 4, 4, 128])

        # Serie di conv2d_transpose con batchnorm e LeakyReLU
        x = conv_transpose(x, [batch_size, scale_h[1], scale_w[1], 128], "G_conv0", initializer=init,
                           strides=[1, 2, 2, 1])
        x = Batchnorm('Generator.BN1', [0, 1, 2], x)
        x = tf.nn.leaky_relu(x)

        x = conv_transpose(x, [batch_size, scale_h[2], scale_w[2], 64], "G_conv1", initializer=init,
                           strides=[1, 2, 2, 1])
        x = Batchnorm('Generator.BN2', [0, 1, 2], x)
        x = tf.nn.leaky_relu(x)

        x = conv_transpose(x, [batch_size, scale_h[3], scale_w[3], 64], "G_conv2", initializer=init,
                           strides=[1, 2, 2, 1])
        x = Batchnorm('Generator.BN3', [0, 1, 2], x)
        x = tf.nn.leaky_relu(x)

        x = conv_transpose(x, [batch_size, scale_h[4], scale_w[4], 32], "G_conv3", initializer=init,
                           strides=[1, 2, 2, 1])
        x = Batchnorm('Generator.BN4', [0, 1, 2], x)
        x = tf.nn.leaky_relu(x)

        x = conv_transpose(x, [batch_size, scale_h[4], scale_w[4], 3], "G_conv4", initializer=init,
                           strides=[1, 1, 1, 1])

        x = tf.tanh(x)

        # Reshape finale per output flattenato (esempio: 64*64*3=12288)
        x = tf.reshape(x, [-1, output_height * output_width * 3])
        return x


def generate_fake_images(generator_output, y_placeholder, z_placeholder, labels, n_samples):
    """
    Genera immagini false utilizzando il generatore.
    Se è la prima volta può servire per caricare modelli pre-addestrati.
    """
    noise_batch = sess.run(sample_noise(n_samples, z_placeholder.shape[1].value or 96))  # fallback 96
    generated_images = sess.run(generator_output,
                                feed_dict={y_placeholder: labels[:n_samples], z_placeholder: noise_batch})
    return generated_images


def save_generator(vars_to_save):
    saver = tf.train.Saver(var_list=vars_to_save)
    saver.save(sess, './my_generator_model.ckpt')
    print("Generatore salvato con successo.")


def load_generator():
    vars_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_conv')
    saver = tf.train.Saver(var_list=vars_to_restore)
    saver.restore(sess, './my_generator_model.ckpt')
    print("Generatore caricato.")


def display_images(images):
    """Visualizza immagini in una griglia quadrata."""
    images = np.reshape(images, [images.shape[0], -1])
    image_number = images.shape[0]
    image_side = int(np.sqrt(images.shape[1]))
    grid_dim = int(np.ceil(np.sqrt(image_number)))

    fig = plt.figure(figsize=(grid_dim, grid_dim))
    gs = gridspec.GridSpec(grid_dim, grid_dim)
    gs.update(wspace=0.05, hspace=0.05)

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(image.squeeze())
    plt.show()


if __name__ == '__main__':
    # --- Esempio di uso ---
    # Definisci placeholder
    z_ph, y_ph = define_placeholders(noise_dim=96, label_dim=10)

    # Costruisci grafo generatore
    generated_sample = generator_conv(z_ph, y_ph, reuse=False)

    # Genera immagini fake (primo batch)
    fake_images = generate_fake_images(generated_sample, y_ph, z_ph, labels, n_samples=128)

    print("Fake images shape:", fake_images.shape)

    # Visualizza prime 25 immagini generate (adatta se vuoi)
    display_images(fake_images[:25])
