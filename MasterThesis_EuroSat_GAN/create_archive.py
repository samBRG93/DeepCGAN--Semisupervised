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


def load_batch():
    sess = tf.Session()
    features, labels = train_input_fn()
    feature, label = sess.run([features['image'], labels])

    

def test_dataset(): 
    sess = tf.Session()
    ## Function to print images start

    ## ------------------------------

    features, labels = train_input_fn()


    # Initialize `iterator` with training data.

    #sess.run(train_iterator.initializer)

    # Initialize `iterator` with validation data.

    #sess.run(val_iterator.initializer)

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

    image = tf.reshape(image, shape=[256,256, 3])

    label = tf.cast(parsed["label"], tf.int32)



    return image, label

def input_fn(filenames, train, batch_size=128, buffer_size=2048):

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
    #return input_fn(filenames=["train_Eurosat.tfrecords", "test_Eurosat.tfrecords"], train=True)

def _int64_feature(value):
    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def print_image(img,title):
    #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) # può essere che debba saltarlo 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
 
    
    
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

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC) # può essere che debba saltarlo 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    '''
    plt.title('con scambio to RGB')
    plt.imshow(img)
    plt.xticks([]), plt.yticks([]) 
    plt.show()
    '''

    return img

 

def createDataRecord(out_filename, addrs, labels):

    # open the TFRecords file



    writer = tf.python_io.TFRecordWriter(out_filename)

    for i in range(len(addrs)): #len(addrs)

        # print how many images are saved every 1000 images

        if not i % 1000:

            print('Train data: {}/{}'.format(i, len(addrs)))

            sys.stdout.flush()

        # Load the image

        img = load_image(addrs[i])
        
        print('addrs: ',addrs[i])


        label = labels[i]
        #print('label: ', label)



        #if img is None:
        #    continue



        # Create a feature

        feature = {

            'image_raw': _bytes_feature(img.tostring()),

            'label': _int64_feature(label)

        }
        
        # Create an example protocol buffer

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        

        # Serialize to string and write on the file

        writer.write(example.SerializeToString())

        

    writer.close()

    sys.stdout.flush()


labels = [] 
labels_train = []
labels_test = []

i = 0 
path_root = []
path_files = []

addrs = []

addrs_train = []
addrs_test = []

for root, dirs, files in os.walk(r'C:\Users\samuele\Desktop\tesi_magistrale\archive\UCMerced_LandUse\Images'):
    
    #print(files)
    length = np.shape(files)[0]
    #print('length: ',length)
    print(root)
    labels.extend(np.full(length, i, dtype=int))
    
    #labels_test.extend(np.full(length-int(0.85*length), i-1, dtype=int))
    #labels_train.extend(np.full(length-int(0.15*length), i-1, dtype=int))  
    
    
    root = root + '\\'
    if i>0:
        for j in range(length):
            addrs.extend([root + files[j]])
            #if j < int(length*0.85) : 
            #    addrs_train.extend([root + files[j]])    
            #if j >= int(length*0.85) : 
            #    addrs_test.extend([root + files[j]]) 
            
    i = i + 1

'''
addrs = np.array(addrs)
labels = np.array(labels)
addrs_train = np.array(addrs_train)
labels_train = np.array(labels_train)
addrs_test = np.array(addrs_test)
labels_test = np.array(labels_test)
'''
'''
labels_test = labels_test - 1 
labels_train = labels_train -1 
labels = labels - 1 
'''

labels[:] = [x - 1 for x in labels]

c = list(zip(addrs, labels))

shuffle(c)

addrs, labels = zip(*c)

train_addrs = addrs[0:int(0.80*len(addrs))]
train_labels = labels[0:int(0.80*len(labels))]

test_addrs = addrs[int(0.80*len(addrs)):]
test_labels = labels[int(0.80*len(labels)):]



#train_path = 'tesi_magistrale/archive/EuroSAT/*/*/*/*/*/*/*/*/*/*.jpg'

# read addresses and labels from the 'train' folder


for i in range(10):
    print("i: ", i)
    print("train_address: ",train_addrs[i])
    print("test address: ",test_addrs[i])

createDataRecord('train_UCmerced.tfrecords', train_addrs, train_labels)
createDataRecord('test_UCmerced.tfrecords', test_addrs, test_labels)



a,b = train_input_fn()
#quelli gia salvati
#train.tfrecords
#test.tfrecords
