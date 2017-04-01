import cv2
import os
import numpy as np
from matplotlib import patches
import data
import matplotlib.pyplot as plt
import network


def run_inference_on_image():

    net = network.Network()
    model = net.get_model()
    model.load('localize_network.net')

    annotations = data.load_annotations()
    filepaths = data.create_image_list(annotations)

    input_vectors, labels = data.get_resized_input_data(filepaths, annotations)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for imagename, input in zip(filepaths, input_vectors):
        #im2 = np.load(data.get_resized_image_path(imagename))
        #ax.imshow(im2)
        #plt.pause(5)
        image = cv2.imread(os.path.join('train/all', imagename))
        im = np.array(image, dtype=np.uint8)
        ax.imshow(im)
        predictions = model.predict([input])[0]

        x = round(predictions[0])
        width = round(predictions[1])
        y = round(predictions[2])
        height = round(predictions[3])

        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.pause(0.5)
        plt.cla()


if __name__ == '__main__':
    run_inference_on_image()




