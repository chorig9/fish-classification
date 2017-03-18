from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import resized_loader

steps = 200
X, Y = resized_loader.get_resized_input_data()

def l2_loss(y_pred, y_true):
    return tf.reduce_sum(tf.square(y_pred - y_true))


# Convolutional network building
network = input_data(shape=[None, 144, 256, 3])
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='linear')
network = regression(network, optimizer='adam',
                     learning_rate=0.001, loss=l2_loss)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)

for step in range(1, steps):
    model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X, Y),
            show_metric=True, batch_size=100, run_id='bounding_box_network')
