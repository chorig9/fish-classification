from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn

class Network:

    def l2_loss(self, y_pred, y_true, x = None):
        with tf.name_scope('L2Accuracy'):
            squared = tf.reduce_sum(tf.square(y_pred - y_true), axis=1)
            l2_distance = tf.sqrt(squared)
            return tf.reduce_mean(l2_distance)

    def accuracy(self, y_pred, y_true, x = None):
        with tf.name_scope('Accuracy'):
            return self.l2_loss(y_pred, y_true) < 100

    def get_model(self):
        input = input_data(shape=[None, None, None, 3])

        size_tensor = tf.constant(value=[224, 224], dtype=tf.int32)
        resized_input = tf.image.resize_bilinear(input, size_tensor)

        network = conv_2d(resized_input, 8, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 8, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 8, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 4, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 4, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = fully_connected(network, 4, activation='linear')
        network = regression(network, optimizer='adam', loss=self.l2_loss,
                             learning_rate=0.001)

        model = tflearn.DNN(network, tensorboard_verbose=0)

        return model