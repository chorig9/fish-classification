from __future__ import division, print_function, absolute_import

import os

import data
import utils
import classifier

size = (122, 122)
X, Y = data.get_cropped_input_data(size, one_hot=True)

X_train,Y_train, X_test, Y_test = utils.split_data(X, Y, 0.1,seed=1337)

net = classifier.Classifier()

# Train using classifier
model = net.get_model(size[0], size[1])

# Load previously trained network snapshot
model.load('classify_network.net')

model.fit(X_train, Y_train, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, run_id='classification_network')

model.save('classify_network.net')