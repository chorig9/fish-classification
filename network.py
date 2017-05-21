from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn

class Network:
    def l2_loss(self, y_pred, y_true, x=None):
        with tf.name_scope('L2Accuracy'):
            squared = tf.reduce_sum(tf.square(y_pred - y_true), axis=1)
            l2_distance = tf.sqrt(squared)
            return tf.reduce_mean(l2_distance)

    def accuracy(self, y_pred, y_true, x=None):
        with tf.name_scope('Accuracy'):
            return self.l2_loss(y_pred, y_true)

    def get_model(self, size_input_1, size_input_2):
        # Convolutional network building
        network = input_data(shape=[None, size_input_1, size_input_2, 3])
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 3)
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4, activation='linear')
        network = regression(network, optimizer='adam',
                             learning_rate=0.001, loss=self.l2_loss, metric=self.accuracy)

        model = tflearn.DNN(network, tensorboard_verbose=0)

        return model