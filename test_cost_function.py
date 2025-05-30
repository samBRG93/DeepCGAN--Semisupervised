import tensorflow as tf
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(BASE_DIR, 'Desktop/working_gan_thesis/EuroSat_GAN'))

from euro_sat_archive import load_batch


def unsupervised_hashing_cost_function(features, hash_vector, alpha=1.0, batch_size=128, p=1.0, d=1.0, k_length=32):
    """
    Calcola la loss per unsupervised hashing.

    features: numpy array, shape (batch_size, feature_dim)
    hash_vector: numpy array, shape (batch_size, k_length)
    alpha: float, peso per la regolarizzazione L1
    p, d: parametri per kernel di similarità
    k_length: lunghezza vettore hash (qui 32)

    Ritorna: loss totale TF tensor e lista delle loss componenti
    """

    # hash_vector_1 e 2 sono metà vettore hash per qualche motivo specifico
    hash_vector_1 = hash_vector[:, :k_length // 2]
    hash_vector_2 = hash_vector[:, k_length // 2:k_length]

    # Calcolo differenze pairwise delta_features tra tutte le coppie nel batch
    # shape: (batch_size, batch_size, feature_dim)
    features_expanded = np.expand_dims(features, axis=1)  # (batch_size,1,feature_dim)
    features_tiled = np.expand_dims(features, axis=0)  # (1,batch_size,feature_dim)
    delta_features = features_expanded - features_tiled  # (batch_size,batch_size,feature_dim)

    # Flatten delta_features per calcoli TF
    delta_features_flat = np.reshape(delta_features, (batch_size * batch_size, features.shape[1]))

    # Conversione hash_vector a tensore TF
    hash_vector_tf = tf.convert_to_tensor(hash_vector, dtype=tf.float32)

    # L1 regolarizzazione verso 1
    l1 = alpha * tf.reduce_mean(tf.abs(hash_vector_tf - 1.0))

    # L2 loss sulla varianza (scostamento medio tra media vettore hash e ogni elemento)
    l2 = tf.nn.l2_loss(tf.reduce_mean(hash_vector_tf, axis=0) - hash_vector_tf)

    # Calcolo similarità s fra coppie usando norma L2 sui delta_features
    delta_features_tf = tf.convert_to_tensor(delta_features_flat, dtype=tf.float32)
    dist = tf.norm(delta_features_tf, axis=1)  # (batch_size*batch_size,)
    s = tf.exp(-dist / (p * d))  # kernel di similarità

    # Calcolo term sim_term fra le due metà del vettore hash (media prodotto scalare normalizzato)
    b_tilde_1 = 2.0 * tf.convert_to_tensor(hash_vector_1, dtype=tf.float32) - 1.0  # (batch_size, k_length/2)
    b_tilde_2 = 2.0 * tf.convert_to_tensor(hash_vector_2, dtype=tf.float32) - 1.0
    sim_term = (tf.reduce_sum(b_tilde_1 * b_tilde_2, axis=1) + k_length / 2) / k_length  # (batch_size,)

    # Ripeti sim_term per confronto con s (dimensione s = batch_size*batch_size)
    sim_term_expanded = tf.reshape(tf.tile(sim_term, [batch_size]), [batch_size * batch_size])

    # L3: perdita somma assoluta differenza similarità calcolate e sim_term
    l3 = tf.reduce_sum(tf.abs(s - sim_term_expanded))

    total_loss = l1 + l2 + l3

    return total_loss, (l1, l2, l3)


if __name__ == '__main__':
    features, labels = load_batch(batch_size=128)  # shape (128, feature_dim)

    # Assumiamo hash_vector estratto da features (ad es. prime 32 dimensioni)
    hash_vector = np.array(features[:, :32], dtype=np.float32)

    loss_op, loss_components = unsupervised_hashing_cost_function(features, hash_vector)

    with tf.Session() as sess:
        total_loss_val, (l1_val, l2_val, l3_val) = sess.run([loss_op, loss_components])
        print(f"Loss L1 (regularization): {l1_val:.6f}")
        print(f"Loss L2 (variance):       {l2_val:.6f}")
        print(f"Loss L3 (similarity):     {l3_val:.6f}")
        print(f"Total loss:               {total_loss_val:.6f}")
