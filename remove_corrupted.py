from matplotlib import patches
import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def remove_images(imagenames):
    os.remove(os.path.join(data.image_dir, imagename))


def onclick(event):
    to_delete.append(imagename)


annotations = data.load_annotations()
filepaths = data.create_image_list(annotations)

input_vectors, labels = data.get_resized_input_data(filepaths, annotations)

imagename = None
to_delete = []

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
for imagename, input in zip(filepaths, input_vectors):
    image = cv2.imread(data.get_image_path(imagename))
    ax.imshow(image)

    predictions = annotations[imagename]
    x = round(predictions[0])
    width = round(predictions[1])
    y = round(predictions[2])
    height = round(predictions[3])

    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.pause(1)
    plt.cla()

print(to_delete)
#remove_images(to_delete)
