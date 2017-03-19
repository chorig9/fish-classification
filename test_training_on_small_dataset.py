import resized_loader
import utils
import numpy as np
import network
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

# Get small dataset
X, Y, imagepaths = resized_loader.get_resized_input_data(ret_filepaths=True)
X,Y, X_test, Y_test, paths = utils.split_data(X, Y, 0.1,seed=1337, fpaths=imagepaths, ret_filepaths=True)

print("Selected dataset shape:", np.shape(X))


# Train on that dataset
net = network.Network()
model = net.get_model()

model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=2, run_id='bounding_box_network')
print("Network has been trained on selected dataset.")

print("Showing results.")
# Evaluate acquired model
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

for image_data, annot, path in zip(X, Y, paths):
    ax.imshow(np.array(Image.open(path.replace('resized', 'all')), dtype=np.uint8))
    
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