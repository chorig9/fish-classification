from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn

class Classifier:

    def get_model(self, width, height):
        input = input_data(shape=[None, width, height, 3])

        network = conv_2d(input, 16, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 16, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 8, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 8, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = fully_connected(network, 7, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001)

        model = tflearn.DNN(network, tensorboard_verbose=0)

        return model