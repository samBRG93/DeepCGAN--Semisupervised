from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, concatenate, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import numpy as np
import cv2
from sklearn.metrics import log_loss
import tensorflow as tf


def preprocess_input(x):
    x = x.astype('float32')
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def conv2d_bn(x, nb_filter, nb_row, nb_col,
              padding='same', strides=(1, 1), use_bias=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(nb_filter, (nb_row, nb_col),
               strides=strides,
               padding=padding,
               use_bias=use_bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def load_cifar10_data(img_rows, img_cols):
    # Usa cifar10 di keras, puoi personalizzare se vuoi
    (X_train, y_train), (X_valid, y_valid) = tf.keras.datasets.cifar10.load_data()

    nb_train_samples = 50  # esempio limitato per velocità
    nb_valid_samples = 10

    X_train = X_train[:nb_train_samples]
    y_train = y_train[:nb_train_samples]
    X_valid = X_valid[:nb_valid_samples]
    y_valid = y_valid[:nb_valid_samples]

    # Ridimensiona immagini
    X_train_resized = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train])
    X_valid_resized = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_valid])

    return X_train_resized, y_train, X_valid_resized, y_valid


def block_inception_a(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    branch_0 = conv2d_bn(input, 96, 1, 1)
    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)
    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)
    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_a(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    branch_0 = conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='valid')
    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2), padding='valid')
    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)
    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_b(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    branch_0 = conv2d_bn(input, 384, 1, 1)
    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)
    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)
    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)
    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_b(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')
    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, strides=(2, 2), padding='valid')
    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)
    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_c(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    branch_0 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)
    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)
    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)
    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def inception_v4_base(input):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    net = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
    net = conv2d_bn(net, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
    branch_1 = conv2d_bn(net, 96, 3, 3, strides=(2, 2), padding='valid')
    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, padding='valid')
    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, padding='valid')
    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, strides=(2, 2), padding='valid')
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
    net = concatenate([branch_0, branch_1], axis=channel_axis)

    for _ in range(4):
        net = block_inception_a(net)

    net = block_reduction_a(net)

    for _ in range(7):
        net = block_inception_b(net)

    net = block_reduction_b(net)

    for _ in range(3):
        net = block_inception_c(net)

    return net


def inception_v4_model(img_rows=299, img_cols=299, color_type=3, num_classes=10, dropout_keep_prob=0.2):
    if K.image_data_format() == 'channels_first':
        inputs = Input(shape=(color_type, img_rows, img_cols))
    else:
        inputs = Input(shape=(img_rows, img_cols, color_type))

    net = inception_v4_base(inputs)

    net = AveragePooling2D(pool_size=(8, 8), padding='valid')(net)
    net = Dropout(dropout_keep_prob)(net)
    net = Flatten()(net)

    predictions = Dense(num_classes, activation='softmax')(net)

    model = Model(inputs, predictions, name='inception_v4')

    weights_path = r'C:\Users\samuele\Desktop\inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights. {e}")

    sgd = SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    img_rows, img_cols = 299, 299
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 10

    X_train, Y_train, X_valid, Y_valid = load_c
