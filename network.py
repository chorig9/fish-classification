from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn

class Network:

    def __init__(self):
        self.input = input_data(shape=[None, None, None, 3])
        self.network = None
        self.bounding_box = None
        self.classifier = None

        self.init_cropping_net()
        self.init_classifier_net()

    def init_cropping_net(self):
        size_tensor = tf.constant(value=[224, 224], dtype=tf.int32)
        resized_input = tf.image.resize_bilinear(self.input, size_tensor)

        self.network = conv_2d(resized_input, 8, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = conv_2d(self.network, 8, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = conv_2d(self.network, 8, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = conv_2d(self.network, 4, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = conv_2d(self.network, 4, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = fully_connected(self.network, 512, activation='relu')
        self.network = self.bounding_box = fully_connected(self.network, 4, activation='linear')
        self.network = regression(self.network, optimizer='adam', loss=self.l2_loss,
                                  learning_rate=0.001)

    def init_classifier_net(self):
        transformed_images = []
        for i in range(tf.shape(self.input)[0]):
            x = self.bounding_box[i, 0]
            y = self.bounding_box[i, 1]
            w = self.bounding_box[i, 2]
            h = self.bounding_box[i, 3]
            transformed_images.append(
                tf.expand_dims(tf.image.crop_to_bounding_box(self.input[i, :, :, :], y, x, h, w), 0))

        self.classifier = tf.concat(0, transformed_images)
        # TODO


    def l2_loss(self, y_pred, y_true, x = None):
        with tf.name_scope('L2Accuracy'):
            squared = tf.reduce_sum(tf.square(y_pred - y_true), axis=1)
            l2_distance = tf.sqrt(squared)
            return tf.reduce_mean(l2_distance)

    def accuracy(self, y_pred, y_true, x = None):
        with tf.name_scope('Accuracy'):
            return self.l2_loss(y_pred, y_true) < 100

    def get_cropping_model(self):
        model = tflearn.DNN(self.network, tensorboard_verbose=0)
        return model

    def get_entire_model(self):
        model = tflearn.DNN(self.classifier, tensorboard_verbose=0)
        return model