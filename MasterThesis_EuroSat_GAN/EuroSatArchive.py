
from random import shuffle

import glob

import sys

import cv2

import numpy as np
import matplotlib.pyplot as plt

#import skimage.io as io

import tensorflow as tf
import os
from os.path import join, getsize


def load_batch(batch_size=128,Train=True):
    sess = tf.Session()
    features, labels = train_input_fn(batch_size,Train)
    labels = tf.one_hot(labels,depth=10)
    features = tf.reshape(features['image'],shape=[-1,12288]) #
    feature, label = sess.run([features, labels])
    feature = feature * (1 / 255.0)
    
   
    return feature, label
    

def test_dataset(): 
    sess = tf.Session()
    ## Function to print images start

    ## ------------------------------

    features, labels = train_input_fn()

    feature, label = sess.run([features['image'], labels])
    #label = label  - 1 
    print(feature.shape, label.shape)

    # Loop over each example in batch
    #img.shape[0]
    feature = feature.astype(np.uint8)
    for i in range(feature.shape[0]): #
        
       #plt.title('Label: ' +str(label[i]))
       title = 'Label: ' +str(label[i])
       print_image(feature[i],title)
    return feature,label

        
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



def input_fn(filenames, train, batch_size=128, buffer_size=2048):
    
    #filenames = filenames[not train] 
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

def train_input_fn(batch_size = 128,Train=True):

    return input_fn(filenames=["train.tfrecords", "test.tfrecords"], train=Train,batch_size=batch_size)

def _int64_feature(value):
    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def print_image(img,title):
    #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) # può essere che debba saltarlo 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
 
    
    img = img.astype(np.uint8)
    #plt.xticks([]), plt.yticks([]) 
    plt.figure(figsize=(3,3))
    plt.title(title)
    plt.imshow(img)
    plt.show()

def load_image(addr):

    # read an image and resize to (224, 224)

    # cv2 load images as BGR, convert it to RGB
    
    img = cv2.imread(addr)

    if img is None:

        return None

    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) # può essere che debba saltarlo 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    plt.title('con scambio to RGB')
    plt.imshow(img)
    plt.xticks([]), plt.yticks([]) 
    plt.show()
    

    return img


a = load_batch(batch_size=10)
b = load_batch(batch_size=10)
