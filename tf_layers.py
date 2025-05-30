import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import join
from random import shuffle

# =========================
#      TFRecord Parsing
# =========================

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[256, 256, 3])

    label = tf.cast(parsed["label"], tf.int32)

    return image, label

# =========================
#     Dataset Pipeline
# =========================

def input_fn(filenames, train=True, batch_size=128, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat()  # Infinite loop
    else:
        dataset = dataset.repeat(1)  # Single epoch

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()

    return {'image': images_batch}, labels_batch


def train_input_fn(batch_size=128, train=True):
    filenames = ["train_UCmerced.tfrecords", "test_UCmerced.tfrecords"]
    return input_fn(filenames=filenames, train=train, batch_size=batch_size)

# =========================
#     Batch Loader
# =========================

def load_batch(batch_size=128, Train=True):
    with tf.Session() as sess:
        features, labels = train_input_fn(batch_size, Train)
        labels = tf.one_hot(labels, depth=10)
        features = tf.reshape(features['image'], shape=[-1, 256 * 256 * 3])

        feature_vals, label_vals = sess.run([features, labels])
        feature_vals = feature_vals / 255.0  # Normalize

    return feature_vals, label_vals

# =========================
#      Visualizzazione
# =========================

def print_image(img, title=''):
    img = img.astype(np.uint8)
    plt.figure(figsize=(3, 3))
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def test_dataset():
    with tf.Session() as sess:
        features, labels = train_input_fn()
        feature_imgs, label_vals = sess.run([features['image'], labels])

        feature_imgs = feature_imgs.astype(np.uint8)

        for i in range(feature_imgs.shape[0]):
            title = 'Label: ' + str(label_vals[i])
            print_image(feature_imgs[i], title)

    return feature_imgs, label_vals

# =========================
#     Image Loader (cv2)
# =========================

def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        print(f"Errore: immagine non trovata in {addr}")
        return None

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.title('Converted to RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    return img

# =========================
#     TFRecord Helpers
# =========================

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
