from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path
from scipy.misc import imread, imresize
import tensorflow as tf
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from keras import optimizer
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(os.path.join(BASE_DIR, 'Desktop\working_gan_thesis\EuroSat_GAN'))


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def batch_all_triplet_loss(labels, embeddings, margin=0.2, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)

    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def triplet_loss(x):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


def reshape_data(x, shape=(-1, 64, 64, 3)):
    return np.reshape(x, shape)


def resize_images(imgs, size=(299, 299)):
    """Ridimensiona tutte le immagini a dimensione 'size'."""
    return np.array([cv2.resize(img, size) for img in imgs])


def prepare_data(batch_size=1000):
    """
    Carica e prepara i dati di training, validation e test.
    Usa la funzione load_batch già definita (che tu hai).
    """
    eurosat_data_train, eurosat_labels_train = load_batch(batch_size=batch_size, train=True)
    eurosat_data_test, eurosat_labels_test = load_batch(batch_size=batch_size // 2, train=False)

    # Reshape dati
    eurosat_data_train = reshape_data(eurosat_data_train)
    eurosat_data_test = reshape_data(eurosat_data_test)

    # Validation set: prendi i primi 200 campioni di train
    eurosat_data_validation = eurosat_data_train[:200]
    eurosat_labels_validation = eurosat_labels_train[:200]

    # Ritaglia train dopo la validazione
    eurosat_data_train = eurosat_data_train[200:]
    eurosat_labels_train = eurosat_labels_train[200:]

    # Resize per InceptionV3
    eurosat_data_train = resize_images(eurosat_data_train)
    eurosat_data_validation = resize_images(eurosat_data_validation)

    return (eurosat_data_train, eurosat_labels_train), (eurosat_data_validation, eurosat_labels_validation), (
    eurosat_data_test, eurosat_labels_test)


def build_model(input_shape=(299, 299, 3), embedding_size=32):
    """
    Costruisce il modello con base InceptionV3 + GlobalAveragePooling + Dense layers.
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    embeddings = Dense(embedding_size, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=embeddings)
    return model, base_model


def freeze_base_layers(base_model):
    """Blocca tutti i layer della base pre-addestrata."""
    for layer in base_model.layers:
        layer.trainable = False


def unfreeze_top_layers(model, num_layers_to_freeze=172):
    """Sblocca solo gli ultimi layers, blocca il resto."""
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[num_layers_to_freeze:]:
        layer.trainable = True


def get_data_generators(train_data, train_labels, val_data, val_labels, batch_size=10):
    """Crea i generatori Keras per train e validazione con data augmentation."""
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
    val_gen = val_datagen.flow(val_data, val_labels, batch_size=batch_size)
    return train_gen, val_gen


def train_top_layers(model, train_gen, val_gen, nb_train_samples, nb_val_samples,
                     epochs=50, batch_size=10, checkpoint_path='cp.top.best.hdf5'):
    """Addestra solo i layers superiori."""
    freeze_base_layers(model.layers[0])  # la base model è il primo layer? meglio passare separatamente base_model

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]

    steps_per_epoch = nb_train_samples // batch_size
    validation_steps = nb_val_samples // batch_size

    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )


def fine_tune_model(model, checkpoint_path, nb_train_samples, nb_val_samples, batch_size=10,
                    epochs=50, num_layers_to_freeze=172, fine_tuned_checkpoint_path='cp.fine_tuned.best.hdf5'):
    """Fine tuning del modello: sblocca top layers e ri-allenamento."""
    if os.path.exists(fine_tuned_checkpoint_path):
        model.load_weights(fine_tuned_checkpoint_path)
        print(f"Checkpoint '{fine_tuned_checkpoint_path}' caricato.")

    unfreeze_top_layers(model, num_layers_to_freeze)

    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(fine_tuned_checkpoint_path, monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]

    train_gen, val_gen = get_data_generators(
        eurosat_data_train, eurosat_labels_train,
        eurosat_data_validation, eurosat_labels_validation,
        batch_size=batch_size
    )

    steps_per_epoch = nb_train_samples // batch_size
    validation_steps = nb_val_samples // batch_size

    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )


def main():
    # Parametri
    nb_train_samples = 2500
    nb_validation_samples = 800
    batch_size = 10
    top_epochs = 50
    fine_tune_epochs = 50

    # 1. Prepara dati
    (eurosat_data_train, eurosat_labels_train), (eurosat_data_validation, eurosat_labels_validation), _ = prepare_data(
        batch_size=1000)

    # 2. Costruisci modello
    model, base_model = build_model()

    # 3. Generatori dati
    train_gen, val_gen = get_data_generators(eurosat_data_train, eurosat_labels_train,
                                             eurosat_data_validation, eurosat_labels_validation,
                                             batch_size=batch_size)

    # 4. Addestra solo top layers
    freeze_base_layers(base_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks_top = [
        ModelCheckpoint('cp.top.best.hdf5', monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]

    model.fit(
        train_gen,
        epochs=top_epochs,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=val_gen,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_top
    )

    # 5. Fine tuning: sblocca top layers e allena di nuovo
    if os.path.exists('cp.fine_tuned.best.hdf5'):
        model.load_weights('cp.fine_tuned.best.hdf5')
        print("Checkpoint fine tuning caricato.")

    unfreeze_top_layers(model, num_layers_to_freeze=172)
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks_fine = [
        ModelCheckpoint('cp.fine_tuned.best.hdf5', monitor='val_accuracy', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]

    model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=val_gen,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_fine
    )

    # 6. Salva pesi finali
    model.save


if __name__ == '__main__':
    main()
