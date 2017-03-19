import numpy as np
import tensorflow as tf
import os

from PIL import Image
from matplotlib import patches
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import network
from resized_loader import *
import tflearn
import cv2

def run_inference_on_image():

    net = network.Network()
    model = net.get_model()
    model.load('localize_network.net')

    filepaths, input_vectors, labels = get_resized_input_data(ret_filepaths=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for n in range(len(filepaths)):
        im2 = np.load(filepaths[n]+".npy")
        ax.imshow(im2)
        plt.pause(5)
        im = np.array(Image.open(filepaths[n].replace('resized', 'all')), dtype=np.uint8)
        ax.imshow(im)
        buf = []
        buf.append(input_vectors[n])
        predictions = model.predict(buf)
        print(predictions)

        predictions = predictions[0]
        x = predictions[0]
        width = predictions[1]
        y = predictions[2]
        height = predictions[3]

        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.pause(0.5)
        plt.cla()


if __name__ == '__main__':
    run_inference_on_image()




