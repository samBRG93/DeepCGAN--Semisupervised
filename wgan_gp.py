import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras import layers, models

# Impostazioni generali
batch_size = 128
noise_dim = 100
image_shape = (64, 64, 3)

# Dataset EuroSat
# Inserisci correttamente il percorso al tuo dataset o modulo
import euro_sat_archive
from euro_sat_archive import load_batch

# ==========================
#      Utility Funzioni
# ==========================

def preprocess(x): return 2. * x - 1.
def deprocess(x): return (x + 1.) / 2.

def sample_noise(batch_size, dim):
    return tf.random.uniform([batch_size, dim], minval=-1.0, maxval=1.0)

def disp_images(images, n=16):
    images = images[:n]
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.imshow(deprocess(img))
    plt.tight_layout()
    plt.show()

# ==========================
#       Generator
# ==========================

def build_generator(noise_dim):
    model = models.Sequential(name='Generator')
    model.add(layers.Dense(4*4*256, input_dim=noise_dim))
    model.add(layers.Reshape((4, 4, 256)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh'))

    return model

# ==========================
#     Discriminator
# ==========================

def build_discriminator():
    model = models.Sequential(name='Discriminator')
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=image_shape))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # logit output

    return model

# ==========================
#         Loss
# ==========================

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# ==========================
#     Training Step
# ==========================

@tf.function
def train_step(generator, discriminator, images, generator_optimizer, discriminator_optimizer):
    noise = sample_noise(batch_size, noise_dim)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# ==========================
#     Main Esecuzione
# ==========================

if __name__ == '__main__':
    generator = build_generator(noise_dim)
    discriminator = build_discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Esempio di un batch
    images, _ = load_batch(batch_size)
    images = preprocess(images)

    # Un singolo step di training
    g_loss, d_loss = train_step(generator, discriminator, images, gen_optimizer, disc_optimizer)

    # Output
    print(f"Generator Loss: {g_loss.numpy():.4f}, Discriminator Loss: {d_loss.numpy():.4f}")

    # Visualizza immagini generate
    noise = sample_noise(16, noise_dim)
    generated = generator(noise, training=False)
    disp_images(generated.numpy())
