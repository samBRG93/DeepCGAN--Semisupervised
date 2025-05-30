import os
import sys
from random import shuffle

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


# --------------------------------------------------
# TFRecord helpers
# --------------------------------------------------

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addr):
    """Carica un'immagine e la prepara per il TFRecord."""
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_data_record(out_filename, addrs, labels):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        if not i % 1000:
            print(f'Saving data: {i}/{len(addrs)}')
            sys.stdout.flush()

        img = load_image(addrs[i])
        if img is None:
            print(f'Warning: Unable to load {addrs[i]}')
            continue

        feature = {
            'image_raw': _bytes_feature(img.tobytes()),
            'label': _int64_feature(labels[i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


# --------------------------------------------------
# Dataset parser and input pipeline
# --------------------------------------------------

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)
    return image, label


def input_fn(filenames, train=True, batch_size=128, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    x = {'image': images_batch}
    y = labels_batch
    return x, y


def train_input_fn():
    return input_fn(filenames=["train_UCmerced.tfrecords", "test_UCmerced.tfrecords"], train=True)


# --------------------------------------------------
# Test dataset visualization
# --------------------------------------------------

def print_image(img, title):
    plt.figure(figsize=(3, 3))
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def test_dataset():
    sess = tf.Session()
    features, labels = train_input_fn()
    feature_batch, label_batch = sess.run([features['image'], labels])
    feature_batch = feature_batch.astype(np.uint8)
    print('Feature shape:', feature_batch.shape)
    print('Label shape:', label_batch.shape)
    for i in range(feature_batch.shape[0]):
        title = f'Label: {label_batch[i]}'
        print_image(feature_batch[i], title)


# --------------------------------------------------
# Main - TFRecord creation
# --------------------------------------------------

if __name__ == '__main__':
    labels = []
    addrs = []
    i = 0

    root_dir = r'C:\Users\samuele\Desktop\tesi_magistrale\archive\UCMerced_LandUse\Images'

    for root, dirs, files in os.walk(root_dir):
        length = len(files)
        print(f"Processing directory: {root} with {length} images.")
        labels.extend(np.full(length, i, dtype=int))
        if i > 0:
            for file in files:
                addrs.append(os.path.join(root, file))
        i += 1

    # Adjust labels to start from 0
    labels = [x - 1 for x in labels]

    # Shuffle data
    combined = list(zip(addrs, labels))
    shuffle(combined)
    addrs, labels = zip(*combined)

    # Train/test split
    split_index = int(0.8 * len(addrs))
    train_addrs, test_addrs = addrs[:split_index], addrs[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Print some examples
    for i in range(min(10, len(train_addrs))):
        print(f"Train sample: {train_addrs[i]}")
    for i in range(min(10, len(test_addrs))):
        print(f"Test sample: {test_addrs[i]}")

    # Create TFRecords
    create_data_record('train_UCmerced.tfrecords', train_addrs, train_labels)
    create_data_record('test_UCmerced.tfrecords', test_addrs, test_labels)

    # Visualize
    test_dataset()
