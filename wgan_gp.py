import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

# --- Configurazione percorso ---
BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(BASE_DIR, 'Desktop/working_gan_thesis/EuroSat_GAN'))

# --- Dataset EuroSat ---
import euro_sat_archive
from euro_sat_archive import load_batch

# --- Parametri di rete ---
critich = 5
batch_size = 128
noise_dim = 100
DIM = 64
o_h, o_w = 64, 64

s_h = (np.array([o_h / 16, o_h / 8, o_h / 4, o_h / 2, o_h])).astype(int)
s_w = (np.array([o_w / 16, o_w / 8, o_w / 4, o_w / 2, o_w])).astype(int)

# --- Visualizzazione ---
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# --- Inizializzazione ---
tf.reset_default_graph()
init = tf.contrib.layers.xavier_initializer()


# =======================
#        Utility
# =======================

def gaussian_noise_layer(input_layer, std=0.2):
    noise = tf.random_normal(tf.shape(input_layer), mean=0.0, stddev=std)
    return input_layer + noise


def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, w) + b


def minibatch_discrimination(input, num_kernels=100, kernel_dim=3):
    x = tf.layers.dense(input, units=num_kernels * kernel_dim)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def preprocess(x): return 2 * x - 1.0


def deprocess(x): return (x + 1.0) / 2.0


def sample_noise(batch_size, dim): return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)


def rel_error(x, y): return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))


def count_params(): return np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])


def session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def disp_images(images):
    N_channels = 3
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.sqrt(images.shape[1] // N_channels))
    fig = plt.figure(figsize=(sqrtn + 2, sqrtn + 2))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.imshow(img.reshape([sqrtimg, sqrtimg, N_channels]))
    plt.show()


# =======================
#      Reti GAN
# =======================

def activation(x, alpha=0.01): return tf.maximum(x, alpha * x)


def get_solvers(lr=3e-4, beta1=0.5, beta2=0.9):
    d_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
    g_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
    return d_solver, g_solver


def generator_conv(x, c, Reuse=True):
    with tf.variable_scope("generator_conv", reuse=Reuse):
        x = tf.concat(axis=1, values=[x, c])
        x = tf.layers.dense(x, units=4 * 4 * 64)
        x = tf.nn.leaky_relu(x)
        x = tf.reshape(x, shape=[-1, 4, 4, 64])
        x = conv_transpose(x, [batch_size, s_h[2], s_w[2], 64], "G_conv0", init, strides=[1, 4, 4, 1])
        x = tf.nn.leaky_relu(x)
        x = conv_transpose(x, [batch_size, s_h[4], s_w[4], 3], "G_conv4", init, strides=[1, 4, 4, 1])
        x = tf.tanh(x)
        return tf.reshape(x, [-1, 4096 * 3])


def discriminator(x, Reuse=True):
    with tf.variable_scope('discriminator', reuse=Reuse):
        keep_prob = 0.5
        x = gaussian_noise_layer(x)
        x = tf.reshape(x, [-1, 64, 64, 3])
        x = conv2d(x, 3, 16, "D_conv0", init, [1, 4, 4, 1])
        x = Batchnorm('Discriminator.BN0', [0, 1, 2], x)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)
        x = conv2d(x, 16, 32, "D_conv1", init, [1, 4, 4, 1])
        x = Batchnorm('Discriminator.BN1', [0, 1, 2], x)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)
        x = minibatch_discrimination(x)
        x = tf.reshape(x, [-1, 4 * 4 * 32])
        classification = tf.layers.dense(x, 10, kernel_initializer=init, name='classification')
        logits = tf.layers.dense(x, 1, kernel_initializer=init, name='logits')
        return logits, x, classification


# =======================
#  Salvataggio Modello
# =======================

def Save_generator(G_vars):
    saver = tf.train.Saver(var_list=G_vars)
    saver.save(sess, 'my_generator_model.ckpt')
    print('Salvato il generatore.')


# =======================
#     Esecuzione test
# =======================
if __name__ == '__main__':
    sess = session()
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)

    minibatch, minibatch_y = load_batch(batch_size=batch_size)
    disp_images(minibatch[0:16])
    print("Finito di stampare immagini del dataset")
