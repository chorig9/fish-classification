from __future__ import division, print_function, absolute_import

import tflearn
import data_loader
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

steps = 200

# Data loading and preprocessing
data_generator = data_loader.get_train_data_for_localization()
X, Y = next(data_generator)

# Real-time data preprocessing
#img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
#img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 100, 100, 3])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='linear')
network = regression(network, optimizer='adam',
                     learning_rate=0.001, loss='mean_square')

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)

for step in range(1, steps):
    model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X, Y),
            show_metric=True, batch_size=100, run_id='cifar10_cnn')