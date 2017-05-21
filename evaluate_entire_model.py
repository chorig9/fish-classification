from matplotlib import patches

import classifier
import numpy as np
import cv2
import tensorflow as tf
import json
import os
import data
import network
import matplotlib.pyplot as plt

predicted_annotations_path = os.path.join(data.workspace, 'annotations', 'predictions.json')

# Saves predicted bounding boxes to json file
def predict_bounding_boxes(model_filename):
    with tf.Graph().as_default():
        net = network.Network()
        model = net.get_model(384, 384)
        model.load(model_filename)

        annotations = data.load_annotations()
        filepaths = data.create_image_list(annotations)

        input_vectors, labels = data.get_resized_input_data(filepaths, annotations)

        predicted_annotations = {}

        for imagename, input in zip(filepaths, input_vectors):
            predictions = model.predict([input])[0]
            predicted_annotations[imagename] = predictions

        with open(predicted_annotations_path, 'w') as data_file:
            json.dump(predicted_annotations, data_file)



def evaluate_classifier(model_filename):
    with tf.Graph().as_default():
        classfier_net = classifier.Classifier()
        classification_model = classfier_net.get_model(122, 122)
        classification_model.load(model_filename)

        annotations = data.load_annotations()
        image_list = data.create_image_list(annotations)

        ok = 0
        n = 0

        with open(predicted_annotations_path) as data_file:
            bounding_box_data = json.load(data_file)

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        for filepath in image_list:

            x = int(bounding_box_data[filepath][0])
            w = int(bounding_box_data[filepath][1])
            y = int(bounding_box_data[filepath][2])
            h = int(bounding_box_data[filepath][3])

            crop = cv2.imread(data.get_image_path(filepath))
            ax.imshow(crop)

            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            crop = crop[y:y + h, x:x + w]
            if crop is None:
                continue
            height, width, _ = crop.shape
            if height == 0 or width == 0:
                continue
            crop = cv2.resize(crop, (122, 122))

            classification = classification_model.predict([crop])[0]

            color = 'red'
            if data.classes[np.argmax(classification)] == data.get_image_label(filepath):
                ok += 1
                color = 'blue'

            n += 1

            ax.text(x + 10, y - 20, data.classes[np.argmax(classification)], bbox={'facecolor': color, 'alpha': 0.5})
            ax.text(0, 0, data.get_image_label(filepath),bbox={'facecolor':'white'})

            plt.pause(0.5)
            plt.cla()


if __name__ == "__main__":
    predict_bounding_boxes('localize_network.net')
    evaluate_classifier('classify_network.net')
