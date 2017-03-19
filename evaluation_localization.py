import numpy as np
from PIL import Image
from matplotlib import patches
import data
import matplotlib.pyplot as plt
import network

def run_inference_on_image():

    #data.resize_images_and_annotations()

    net = network.Network()
    model = net.get_model()
    model.load('localize_network.net')

    filepaths = data.create_image_list()

    input_vectors, labels = data.get_resized_input_data(filepaths)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for imagename, input in zip(filepaths, input_vectors):
        im2 = np.load(data.get_resized_image_path(imagename))
        ax.imshow(im2)
        plt.pause(5)
        im = np.array(Image.open(data.get_image_path(imagename)), dtype=np.uint8)
        ax.imshow(im)
        predictions = model.predict([input])
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




