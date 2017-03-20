import json
import os
import cv2
import numpy as np
from tensorflow.python.platform import gfile

workspace = os.path.dirname(__file__)
original_annotation_path = os.path.join(workspace, 'annotations', 'all.json')
box_annotations_path = os.path.join(workspace, 'annotations', 'resized_all.json')
image_dir = os.path.join(workspace, 'train', 'all')
resized_output = os.path.join(workspace, 'train', 'resized')


def create_annotations():
    """
    Returns:
        dictionary containing annotation box for every filename
    """
    with open(original_annotation_path) as data_file:
        bounding_box_data = json.load(data_file)

    annotations = {}
    for element in bounding_box_data:
        filename = os.path.basename(element['filename'])

        xstart = element['annotations'][0]['x']
        width = element['annotations'][0]['width']
        ystart = element['annotations'][0]['y']
        height = element['annotations'][0]['height']
        annotations[filename] = [xstart, width, ystart, height]

    return annotations


def load_annotations():
    """
    Returns:
        dictionary containing list [xstart, width, ystart, height] for every filename
    """
    with open(box_annotations_path) as data_file:
        return json.load(data_file)


def create_image_list(annotations):
    """
    Args:
        annotations: dictionary returend by load_annotations()
    Returns:
        list of image names (which have annotations)
    """
    extension = 'jpg'
    image_list = []
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list = gfile.Glob(file_glob)

    for file_name in file_list:
        if os.path.basename(file_name) in annotations.keys():
            image_list.append(os.path.basename(file_name))
    return image_list


def resize_images_and_annotations(image_list, annotations):
    """
    Overrides original images with resized ones, and creates proper annotations.
    Args:
        image_list: images names to be resized
        annotations: dictionary with [x, width, y, height] for every file
    """
    for image_name in image_list:
        path = get_image_path(image_name)
        image = cv2.imread(path)
        im_height, im_width, _ = image.shape

        scale_x = 1280.0 / im_width
        scale_y = 720.0 / im_height

        basename = os.path.basename(image_name)
        x = annotations[basename][0]
        width = annotations[basename][1]
        y = annotations[basename][2]
        height = annotations[basename][3]

        annotations[basename] = [x * scale_x, width * scale_x, y * scale_y, height * scale_y]

        image = cv2.resize(image, (1280, 720))
        cv2.imwrite(path, image)

    with open(box_annotations_path, 'w') as data_file:
        json.dump(annotations, data_file)


def resize_images_to_npy(width, height, image_list):
    """
    Resizes images and saves them to 'resized_output'
    Args:
        width: of target image
        height: of target image
        image_list: images to resize

    """
    for image_name in image_list:
        path = get_image_path(image_name)
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))

        resized_path = get_resized_image_path(image_name)
        np.save(resized_path, image)


def get_resized_input_data(image_list, annotations):
    """
    Args:
        image_list: image names to be loaded
        annotations: dictionary with [x, width, y, height] for every file

    Returns:
        list of resized images and lsit of corresponding annotations
    """
    X = []
    Y = []
    for image in image_list:
        path = os.path.join(resized_output, image)

        X.append(np.load(path + '.npy'))
        Y.append(annotations[image])

    return X, Y


def get_resized_image_path(imagename):
    return os.path.join(resized_output, imagename) + '.npy'


def get_image_path(imagename):
    return os.path.join(image_dir, imagename)


if __name__ == "__main__":
    annotations = create_annotations()
    image_list = create_image_list(annotations)
    resize_images_and_annotations(image_list, annotations)
    resize_images_to_npy(256, 144, image_list)
