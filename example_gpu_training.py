#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
#
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
#
import numpy as np

nsize = 32
batch_size = 32
#
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
x_train = np.expand_dims(x_train, axis=-1)
x_train = np.repeat(x_train, 3, axis=-1)
x_train = x_train.astype('float32') / 255
x_train = tf.image.resize(x_train, [nsize,nsize])
#
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
#
x_test = np.expand_dims(x_test, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
x_test = x_test.astype('float32') / 255
x_test = tf.image.resize(x_test, [nsize,nsize])
#
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 10)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
#
# Open a strategy scope
with strategy.scope():
    """
    Load the Model
    """
    #
    input_tensor = Input(shape=(nsize,nsize,3))
    #
    stem_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=None,
        classifier_activation=None,
        include_preprocessing=True
    )
    #
    flatten = Flatten()(stem_model.output)
    dense1 = Dense(4080, activation='relu')(flatten)
    dense2 = Dense(4080, activation='relu')(dense1)
    output_tensor = Dense(10,activation='softmax')(dense2)
    #
    model = Model(input_tensor, output_tensor)
    #
    model.summary()
    """
    Compile the Model, dependent on the loss function defined in custom_utils.py
    """
    #
    model.compile(
        optimizer = Adam(learning_rate=0.0001),
        loss = CategoricalCrossentropy(),
        metrics = ["acc"]
    )

model.fit(x_train, y_train, epochs=1)
