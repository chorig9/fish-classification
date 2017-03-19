from __future__ import division, print_function, absolute_import

import data
from utils import *
import network

annotations = data.load_annotations()
filepaths = data.create_image_list(annotations)

X, Y = data.get_resized_input_data(filepaths, annotations)
X, Y, X_test, Y_test = split_data(X, Y, 0.1)

net = network.Network()

# Train using classifier
model = net.get_model()

# Load previously trained network snapshot
#model.load('localize_network.net')

model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=50, run_id='bounding_box_network')

model.save('localize_network.net')

