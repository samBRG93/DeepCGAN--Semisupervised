import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Config GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === Hyperparameters ===
BATCH_SIZE = 128
NOISE_DIM = 128
IMAGE_SIZE = 64
CHANNELS = 3


# === Utility Functions ===
def sample_noise(batch_size, dim):
    return tf.random.uniform(shape=[batch_size, dim], minval=-1, maxval=1)


def preprocess(x):
    return (x / 127.5) - 1.0


def deprocess(x):
    return (x + 1.0) * 127.5


def disp_images(images, cols=4):
    images = deprocess(images.numpy()).astype(np.uint8)
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.remove()
    plt.tight_layout()
    plt.show()


def get_batch(data, labels, iteration, batch_size=128):
    total = len(data)
    start = (iteration * batch_size) % total
    end = start + batch_size
    if end > total:
        return data[-batch_size:], labels[-batch_size:]
    return data[start:end], labels[start:end]


def activation(x, alpha=0.01):
    return tf.maximum(x, alpha * x)


# === Batch Normalization Layer ===
def batch_norm(inputs, training):
    return layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs, training=training)


# === Noise Layer ===
def gaussian_noise(inputs, std=0.2):
    return inputs + tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=std)


# === Save JPEG images ===
def save_images(images, iteration, save_dir="saved_images", last=False):
    os.makedirs(save_dir, exist_ok=True)
    images = deprocess(images).numpy().astype(np.uint8)
    for i, img in enumerate(images[:32 if last else 16]):
        tf.keras.utils.save_img(
            os.path.join(save_dir, f"iter{iteration}_img{i}.jpeg"),
            img
        )


# === Confusion Matrix Helper ===
def eval_confusion_matrix(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=10)
    return cm


import tensorflow as tf
import numpy as np


# === Ottimizzatori per Discriminatore e Generatore ===
def get_solvers(lr=3e-4, beta1=0.5, beta2=0.9):
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
    return d_optimizer, g_optimizer


# === Huber Loss ===
def huber_loss(labels, predictions, delta=1.0):
    error = tf.abs(predictions - labels)
    is_small = error < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (error - 0.5 * delta)
    return tf.where(is_small, squared_loss, linear_loss)


# === Differenze casuali tra batch (maschera binaria) ===
def find_diff_batch(batch):
    batch = np.array(batch)
    y_shape, x_shape = batch.shape
    diff_batch = np.zeros_like(batch)

    zero_indices = np.where(batch == 0)
    zero_x = zero_indices[1].reshape(-1, x_shape - 1)

    chosen = np.random.randint(0, x_shape - 1)
    posx = zero_x[:, chosen]
    posy = np.arange(len(posx))

    diff_batch[posy, posx] = 1
    return diff_batch


# === Strato convolutivo ===
def conv2d(x, filters, kernel_size=5, strides=1, padding='same', name=None):
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                  padding=padding, activation=None,
                                  kernel_initializer='glorot_uniform', name=name)(x)


# === Transposed Convolution ===
def conv_transpose(x, filters, kernel_size=5, strides=2, padding='same', name=None):
    return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides,
                                           padding=padding, activation=None,
                                           kernel_initializer='glorot_uniform', name=name)(x)


# === Generatore (Architettura CNN) ===
def build_generator(latent_dim, cond_dim, output_shape=(64, 64, 3)):
    z_input = tf.keras.Input(shape=(latent_dim,))
    c_input = tf.keras.Input(shape=(cond_dim,))

    x = tf.keras.layers.Concatenate()([z_input, c_input])
    x = tf.keras.layers.Dense(4 * 4 * 128, activation='relu')(x)
    x = tf.keras.layers.Reshape((4, 4, 128))(x)

    x = tf.keras.layers.BatchNormalization()(conv_transpose(x, 64, strides=2))
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(conv_transpose(x, 32, strides=2))
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.BatchNormalization()(conv_transpose(x, 16, strides=2))
    x = tf.keras.layers.LeakyReLU()(x)

    x = conv_transpose(x, output_shape[-1], strides=2)
    output = tf.keras.layers.Activation('tanh')(x)

    model = tf.keras.Model(inputs=[z_input, c_input], outputs=output, name="Generator")
    return model


def discriminator(x, c, Reuse=True, ):
    with tf.variable_scope('discriminator', reuse=Reuse):
        keep_prob = 0.5
        init = tf.contrib.layers.xavier_initializer()  # ☻canbiare

        x = gaussian_noise_layer(x)
        x = tf.reshape(x, [-1, 64, 64, 3])

        x = (conv2d(x, 3, 32, "D_conv0", initializer=init, strides=[1, 4, 4, 1]))
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        x = (conv2d(x, 32, 32, "D_conv1", initializer=init, strides=[1, 4, 4, 1]))
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        x = (conv2d(x, 64, 64, "D_conv2", initializer=init, strides=[1, 2, 2, 1]))
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        x = (conv2d(x, 64, 64, "D_conv3", initializer=init, strides=[1, 2, 2, 1]))
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dropout(x, keep_prob)

        x = tf.reshape(x, [-1, 4 * 4 * 32])
        feature_match = x
        print("x shape: ", x.shape)
        print("c shape: ", c.shape)

        x = tf.concat(values=[x, c], axis=1)

        hash_stream = tf.nn.sigmoid(tf.layers.dense(x, units=32, kernel_initializer=init, activation=None))

        classification = tf.layers.dense(x, units=10, kernel_initializer=init, activation=None)
        logits = tf.layers.dense(x, 1, kernel_initializer=init, name='logits')

        return logits, feature_match, classification, hash_stream


def save_generator(G_vars):
    saver = tf.train.Saver(var_list=G_vars)
    saver.save(sess, 'my_generator_model.ckpt')
    print('salvo il generatore')


def load_generator():
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_conv'))
    saver.restore(session, 'my_generator_model.ckpt')


def train(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,
          show_every=250, print_every=50, batch_size=128, num_epoch=10):
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    import numpy as np

    # Configurazione TensorBoard
    tf.summary.scalar("Discriminator loss", D_loss)
    tf.summary.scalar("Generator loss", G_loss)
    merged_summary_op = tf.summary.merge_all()
    logs_path = '/tmp/tensorflow_logs/example/'
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    N_EuroSatTrain_ex = 22950
    max_iter = int(N_EuroSatTrain_ex * num_epoch / batch_size)
    print(f"Max iterations: {max_iter}")

    for it in range(max_iter):
        # Aggiorna il discriminatore 'critich' volte (assumendo critich definito globalmente)
        for _ in range(critich):
            minibatch, minibatch_y = load_batch(batch_size=batch_size)
            Neg_Batch = Find_diff_batch(minibatch_y)
            _, D_loss_curr = sess.run([D_train_step, D_loss],
                                      feed_dict={x: minibatch, y: minibatch_y, Neg_Cond: Neg_Batch, D_t: 1, T_t: 0})
            # Esegui eventuali update ops extra per il discriminatore
            sess.run(D_extra_step)

        # Aggiorna il generatore
        _, G_loss_curr, summary = sess.run([G_train_step, G_loss, merged_summary_op],
                                           feed_dict={x: minibatch, y: minibatch_y, Neg_Cond: Neg_Batch, D_t: 1,
                                                      T_t: 0})
        # Esegui eventuali update ops extra per il generatore
        sess.run(G_extra_step)

        # Scrivi summary per TensorBoard
        summary_writer.add_summary(summary, it)

        # Stampa info ogni print_every iterazioni
        if it % print_every == 0:
            # Valutazione accurata su test set (eurosat_data_test, eurosat_labels_test devono essere definiti)
            logits_test = sess.run(discriminator(preprocess(x), tf.zeros_like(eurosat_labels_test))[-2],
                                   # C_real logits
                                   feed_dict={x: eurosat_data_test})
            pred_classes = np.argmax(logits_test, axis=1)
            true_classes = np.argmax(eurosat_labels_test, axis=1)

            target_names = [f'class {i}' for i in range(y_dim)]
            print(f"Iter {it} - D_loss: {D_loss_curr:.4f}, G_loss: {G_loss_curr:.4f}")
            print("Classification Report:\n",
                  classification_report(true_classes, pred_classes, target_names=target_names))

            correct = np.equal(pred_classes, true_classes)
            accuracy = np.mean(correct)
            print(f"Accuracy on test set: {accuracy:.4f}")

        # Visualizza immagini generate ogni show_every iterazioni
        if it % show_every == 0:
            samples = sess.run(G_sample, feed_dict={y: minibatch_y})
            samples = deprocess(samples)
            fig = disp_images(samples[:16])
            plt.show()

    # Dopo addestramento, genera immagini condizionate per ogni classe
    print("Generazione immagini condizionate per ciascuna classe...")
    for i in range(y_dim):
        c = np.zeros((batch_size, y_dim), dtype=np.float32)
        c[:, i] = 1
        samples = sess.run(G_sample, feed_dict={y: c})
        samples = deprocess(samples)
        fig = disp_images(samples[:16])
        plt.show()

    # Salva modello
    print("Salvataggio modello...")
    save_path = saver.save(sess, "/tmp/EuroSat_complete_model.ckpt")
    print(f"Model saved in path: {save_path}")

    return samples



def soft_noisy_loss(real_discriminator, fake_discriminator, batch_size=128, noise_prop=0.05):
    # Crea etichette morbide e rumorose per dati reali (vicine a 1) e fake (vicine a 0)

    # True labels: vicine a 1 con un po' di rumore (0.9-1.0)
    true_labels = 1.0 - tf.random.uniform([batch_size, 1], minval=0.0, maxval=0.1)

    # Flip randomico di una piccola percentuale di etichette (5%)
    flip_mask = tf.random.uniform([batch_size, 1]) < noise_prop
    true_labels = tf.where(flip_mask, 1.0 - true_labels, true_labels)

    # Discriminator loss su dati reali
    d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_discriminator, labels=true_labels))

    # Fake labels: vicine a 0 con un po' di rumore (0.0-0.1)
    fake_labels = tf.random.uniform([batch_size, 1], minval=0.0, maxval=0.1)
    flip_mask = tf.random.uniform([batch_size, 1]) < noise_prop
    fake_labels = tf.where(flip_mask, 1.0 - fake_labels, fake_labels)

    # Discriminator loss su dati generati
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_discriminator, labels=fake_labels))

    # Loss totale del discriminatore (media)
    d_loss = 0.5 * tf.add(d_loss_true, d_loss_fake)

    # Loss del generatore (vuole ingannare il discriminatore, quindi etichette "vere")
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_discriminator, labels=tf.ones_like(fake_discriminator)))

    return d_loss, g_loss


def wgan_loss(logits_real, logits_fake, batch_size, x, G_sample, discriminator, lam=10.0):
    # Loss WGAN-GP con penalità del gradiente

    d_loss = -tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
    g_loss = -tf.reduce_mean(logits_fake)

    # Interpolazione per il calcolo del gradiente
    eps = tf.random.uniform([batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    x_hat = eps * x + (1.0 - eps) * G_sample

    # Calcolo discriminatore su x_hat (con riuso variabili)
    with tf.variable_scope("", reuse=True):
        disc_x_hat, _, _, _ = discriminator(x_hat)

    # Gradiente rispetto a x_hat
    grad = tf.gradients(disc_x_hat, [x_hat])[0]
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]) + 1e-12)  # norma euclidea per immagine batch

    # Penalità gradiente (GP)
    grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))

    # Aggiungi penalità gradiente alla loss discriminatore
    d_loss += lam * grad_penalty

    return d_loss, g_loss


def lossless_triplet_loss(anchor, positive, negative, N=32, beta=32, epsilon=1e-8):
    """
    Lossless triplet loss function.

    Arguments:
    anchor -- tensor, embeddings for anchor samples
    positive -- tensor, embeddings for positive samples (similar to anchor)
    negative -- tensor, embeddings for negative samples (different from anchor)
    N -- int, recommended dimension (scaling factor)
    beta -- float, scaling factor for distances
    epsilon -- float, small value to avoid log(0)

    Returns:
    loss -- scalar tensor, average triplet loss over the batch
    """

    # Compute squared distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Apply non-linear transformations with stable log arguments
    pos_term = -tf.math.log(tf.maximum(1 + epsilon - (pos_dist / beta), epsilon))
    neg_term = -tf.math.log(tf.maximum(1 + epsilon - ((N - neg_dist) / beta), epsilon))

    loss = pos_term + neg_term

    # Return mean loss over batch
    return tf.reduce_mean(loss)


if __name__ == "__main__":
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
    h = tf.placeholder(tf.float32, shape=[None, y_dim])
    x = tf.placeholder(tf.float32, [None, 4096 * 3])
    z = sample_noise(batch_size, noise_dim)

    # Placeholder per condizione negativa
    Neg_Cond = tf.placeholder(tf.float32, shape=[None, y_dim])

    # Generatore produce campioni condizionati
    G_sample = generator_conv(z, y, Reuse=False)
    G_sample_noisy = gaussian_noise_layer(G_sample)

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        # Discriminatore su dati reali pre-processati
        logits_real, feature_real, C_real, Hash_anchor = discriminator(preprocess(x), tf.zeros_like(y), Reuse=False)

        # Se vuoi riusare C_real come input condizionale, fallo esplicitamente
        # Qui sembra un passaggio ridondante o errore, se vuoi puoi eliminarlo o sostituirlo
        # logits_real, feature_real, C_real, _ = discriminator(preprocess(x), C_real)

        # Discriminatore su dati fake (noisy)
        logits_fake, feature_fake, C_fake, Hash_positive = discriminator(G_sample_noisy, tf.zeros_like(y))

        # Se vuoi riusare C_fake come input condizionale:
        # logits_fake, feature_fake, C_fake, _ = discriminator(G_sample_noisy, c=C_fake)

    # Generatore produce immagini negative condizionate
    Negative_fake_image = generator_conv(z, Neg_Cond)

    # Discriminatore su immagini negative generate
    __, _, _, Hash_negative = discriminator(Negative_fake_image, tf.zeros_like(y))

    # Variabili da ottimizzare
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_conv')

    # Ottimizzatori
    d_solver, g_solver = get_solvers()

    # Calcolo loss
    d_loss, g_loss = soft_noisy_loss(logits_real, logits_fake)

    # Loss triplet
    triplet_loss = lossless_triplet_loss(y, Hash_anchor, Hash_positive, Hash_negative, N=32, beta=32, epsilon=1e-8)

    # Loss classificazione
    print('logits shape: ', np.shape(C_real))
    s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_real, labels=y))
    d_loss += s_loss

    g_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_fake, labels=y))

    # Image matching loss (huber)
    recon_weight = tf.constant(1.0, dtype=tf.float32)
    g_loss += tf.reduce_mean(
        huber_loss(
            tf.reshape(preprocess(x), [-1, 64 * 64 * 3]),
            tf.reshape(G_sample, [-1, 64 * 64 * 3])
        )
    ) * recon_weight

    # Feature matching (L2 between features)
    tmp1 = tf.reduce_mean(feature_real, axis=0)
    tmp2 = tf.reduce_mean(feature_fake, axis=0)
    G_L2 = tf.reduce_mean(tf.square(tmp1 - tmp2))
    g_loss += G_L2

    # TODO: Definisci D_t e T_t come placeholder o costanti
    # Esempio placeholder:
    # D_t = tf.placeholder(tf.float32, shape=[])
    # T_t = tf.placeholder(tf.float32, shape=[])

    # Moltiplico le loss (usa 1.0 se non usi maschere)
    triplet_loss = tf.multiply(T_t, triplet_loss)
    d_loss = tf.multiply(D_t, d_loss)
    d_loss = triplet_loss + d_loss

    # Ottimizzatori con update ops
    d_train_step = d_solver.minimize(d_loss, var_list=D_vars)
    g_train_step = g_solver.minimize(g_loss, var_list=G_vars)

    d_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    g_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_conv')

    with tf.Session() as sess:
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        samples = train(
            sess,
            g_train_step,
            g_loss,
            d_train_step,
            d_loss,
            g_extra_step,
            d_extra_step,
            batch_size=batch_size,
            num_epoch=40
        )

        save_generator(G_vars)
