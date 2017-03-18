from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn

class Network:

    def l2_loss(self, y_pred, y_true, x = None):
        with tf.name_scope('L2Accuracy'):
            return tf.nn.l2_loss(y_pred - y_true)

    def accuracy(self, y_pred, y_true, x = None):
        with tf.name_scope('InsideAccuracy'):
            conditions = [False, True, False, True]
            a = tf.where(conditions, tf.matrix_transpose(y_pred), tf.matrix_transpose(y_true))
            b = tf.where(not conditions, tf.matrix_transpose(y_pred), tf.matrix_transpose(y_true))
            ok = tf.reduce_all(tf.greater(a, b), axis=0)
            return tf.cast(tf.count_nonzero(ok), tf.float32)

    def get_model(self):
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
                             learning_rate=0.001, loss=self.l2_loss, metric=self.accuracy)

        model = tflearn.DNN(network, tensorboard_verbose=0)

        return model