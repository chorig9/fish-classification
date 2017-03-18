import numpy as np
import cv2
import os
import image_list

workspace = os.path.dirname(__file__)
resized_output = os.path.join(workspace, 'train', 'resized')


def resize_images(image_list, output_dir=resized_output):
    print(output_dir)
    for image_name in image_list:
        image = cv2.imread(image_name)
        image = cv2.resize(image, (256, 144))

        path = os.path.join(output_dir, os.path.basename(image_name)) + '.npy'
        np.save(path, image)


def get_resized_input_data(image_dir=resized_output, annotations=image_list.create_annotations()):
    X = []
    Y = []
    image_list = os.listdir(image_dir)

    for image in image_list:
        path = os.path.join(image_dir, image)
        name = image[:-4] #remove .npy

        X.append(np.load(path))
        Y.append(annotations[name])

    return X, Y

#resize_images(image_list.create_image_list())
