import cv2
import data
import utils
import numpy as np
import network
import matplotlib.pyplot as plt
from matplotlib import patches

annotations = data.load_annotations()
images_list = data.create_image_list(annotations)

X, Y = data.get_resized_input_data(images_list, annotations)

# get indexes for train and test data
test, train = utils.random_split_indexes(20)

X_test = X[test]
X_train = X[train]

Y_test = Y[test]
Y_train = Y[train]

test_images = images_list[test]

print("Selected dataset shape:", np.shape(X_train))
# Train on that dataset
net = network.Network()
model = net.get_model()

model.fit(X_train, Y_train, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=2, run_id='bounding_box_network')
print("Network has been trained on selected dataset.")

print("Showing results.")
# Evaluate acquired model
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

for image_data, annot, test_image in zip(X, Y, test_images):
    image = cv2.imread(data.get_image_path(test_image))
    ax.imshow(image)
    
    predictions = model.predict([image_data])
    print(predictions)

    predictions = predictions[0]
    width = predictions[1]
    height = predictions[3]
    x = predictions[0]-0.1*width
    y = predictions[2]-0.1*height
    width = width * 1.2
    height = height * 1.2
    

    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    predictions = annot
    width = predictions[1]
    height = predictions[3]
    x = predictions[0]-0.1*width
    y = predictions[2]-0.1*height
    width = width * 1.2
    height = height * 1.2
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    plt.pause(0.5)
    plt.cla()