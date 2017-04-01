from __future__ import division, print_function, absolute_import

import os

import data
import utils
import network

annotations = data.load_annotations()
filepaths = data.create_image_list(annotations)

X, Y = data.get_resized_input_data(filepaths, annotations)

X_train,Y_train, X_test, Y_test = utils.split_data(X, Y, 0.1,seed=1337, ret_filepaths=False)

net = network.Network()

# Train using classifier
model = net.get_model(384, 384)

# Load previously trained network snapshot
#model.load('localization_network.net')

model.fit(X_train, Y_train, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=50, run_id='bounding_box_network')

os.chdir(data.localization_dir)
model.save('localize_network.net')