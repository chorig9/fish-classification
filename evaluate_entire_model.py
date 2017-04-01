import classifier
import data
import numpy as np
import cv2
import tensorflow as tf
import json
import os
import data
import network

predicted_annotations_path = os.path.join(data.workspace, 'annotations', 'predictions.json')

def predict_bounding_boxes():
    with tf.Graph().as_default():
        net = network.Network()
        model = net.get_model()
        model.load('localize_network.net')

        annotations = data.load_annotations()
        filepaths = data.create_image_list(annotations)

        input_vectors, labels = data.get_resized_input_data(filepaths, annotations)

        predicted_annotations = {}

        for imagename, input in zip(filepaths, input_vectors):
            predictions = model.predict([input])[0]

            x = round(predictions[0])
            width = round(predictions[1])
            y = round(predictions[2])
            height = round(predictions[3])

            predicted_annotations[imagename] = predictions

        with open(predicted_annotations_path, 'w') as data_file:
            json.dump(predicted_annotations, data_file)


def evaluate_classifier():
    with tf.Graph().as_default():
        classfier_net = classifier.Classifier()
        classification_model = classfier_net.get_model(122, 122)
        classification_model.load('classify_network.net')

        annotations = data.load_annotations()
        image_list = data.create_image_list(annotations)

        ok = 0
        n = 0

        with open(predicted_annotations_path) as data_file:
            bounding_box_data = json.load(data_file)

        for filepath in image_list:
            x = bounding_box_data[filepath][0]
            w = bounding_box_data[filepath][1]
            y = bounding_box_data[filepath][2]
            h = bounding_box_data[filepath][3]

            crop = cv2.imread(data.get_image_path(filepath))
            crop = crop[y:y + h, x:x + w]
            if crop is None:
                continue
            height, width, _ = crop.shape
            if height == 0 or width == 0:
                continue
            crop = cv2.resize(crop, (122, 122))

            classification = classification_model.predict([crop])[0]

            if data.classes[np.argmax(classification)] == data.get_image_label(filepath):
                ok += 1

            print(data.classes[np.argmax(classification)] + " " + data.get_image_label(filepath))

            n += 1

        print(ok / n)


if __name__ == "__main__":
   # predict_bounding_boxes()
    evaluate_classifier()
