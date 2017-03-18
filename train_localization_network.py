from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import tensorflow as tf
import resized_loader
from utils import *
import network

X, Y = resized_loader.get_resized_input_data()
X, Y, X_test, Y_test = split_data(X, Y, 0.1)

net = network.Network()

# Train using classifier
model = tflearn.DNN(net.get_model(), tensorboard_verbose=0)


model.fit(X, Y, n_epoch=4, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, run_id='bounding_box_network')

model.save('localize_network.net')

