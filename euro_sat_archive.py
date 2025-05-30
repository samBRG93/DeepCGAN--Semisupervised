import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }

    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[64, 64, 3])
    label = tf.cast(parsed["label"], tf.int32)

    return image, label

def input_fn(filenames, train=True, batch_size=128, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size).repeat()
    else:
        dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    return {'image': images_batch}, labels_batch

def train_input_fn(batch_size=128, Train=True):
    return input_fn(["train.tfrecords", "test.tfrecords"], train=Train, batch_size=batch_size)

def load_batch(batch_size=128, train=True):
    with tf.Session() as sess:
        features, labels = train_input_fn(batch_size, train)
        labels = tf.one_hot(labels, depth=10)
        images = tf.reshape(features['image'], shape=[-1, 12288])
        feature_vals, label_vals = sess.run([images, labels])
        feature_vals = feature_vals / 255.0
        return feature_vals, label_vals

def print_image(img, title):
    img = img.astype(np.uint8)
    plt.figure(figsize=(3, 3))
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def test_dataset():
    with tf.Session() as sess:
        features, labels = train_input_fn(batch_size=10, Train=False)
        image_vals, label_vals = sess.run([features['image'], labels])
        image_vals = image_vals.astype(np.uint8)
        for i in range(image_vals.shape[0]):
            print_image(image_vals[i], f'Label: {label_vals[i]}')
        return image_vals, label_vals

def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        print(f"Errore nel caricamento dell'immagine: {addr}")
        return None
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title('Immagine RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

# Esempio di uso
if __name__ == "__main__":
    features, labels = load_batch(batch_size=10)
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    test_dataset()
