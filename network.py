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
        with tf.name_scope('InsideAccuracy'):
            conditions = [False, True, False, True]
            a = tf.where(conditions, tf.matrix_transpose(y_pred), tf.matrix_transpose(y_true))
            b = tf.where(not conditions, tf.matrix_transpose(y_pred), tf.matrix_transpose(y_true))
            ok = tf.reduce_all(tf.greater(a, b), axis=0)
            return tf.cast(tf.count_nonzero(ok), tf.float32)

    def get_model(self):
        # Convolutional network building
        network = input_data(shape=[None, 144, 256, 3])
        network = conv_2d(network, 64, 2, activation='relu')
        network = conv_2d(network, 16, 2, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 16, 2, activation='relu')
        network = conv_2d(network, 8, 2, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 4, 2, activation='relu')
        network = fully_connected(network, 256, activation='relu')
        # network = dropout(network, 0.5)
        network = fully_connected(network, 4, activation='linear')
        network = regression(network, optimizer='adam',
                             learning_rate=0.0001, loss=self.l2_loss, metric=self.accuracy)

        model = tflearn.DNN(network, tensorboard_verbose=0)

        return model