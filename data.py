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
cropped_output = os.path.join(workspace, 'train', 'cropped')

classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']


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


def resize_images_and_annotations():
    """
    Overrides original images with resized ones, and creates proper annotations.
    """

    annotations = create_annotations()
    image_list = create_image_list(annotations)

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


def resize_images_to_npy(width, height):
    """
    Resizes images and saves them to 'resized_output'
    Args:
        width: of target image
        height: of target image

    """

    annotations = load_annotations()
    image_list = create_image_list(annotations)

    for image_name in image_list:
        path = get_image_path(image_name)
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))

        resized_path = get_resized_image_path(image_name)
        np.save(resized_path, image)


def get_resized_input_data(image_list, annotations):
    """
    Returns:
        list of resized images and list of corresponding annotations
    """

    X = []
    Y = []
    for image in image_list:
        path = os.path.join(resized_output, image)

        X.append(np.load(path + '.npy'))
        Y.append(annotations[image])

    return X, Y


def crop_images(image_list, bounding_box_data):
    """
    Args:
        image_list: image names to be loaded
        bounding_box_data: coordinates of crop
    """

    for image in image_list:
        x = int(bounding_box_data[image][0])
        w = int(bounding_box_data[image][1])
        y = int(bounding_box_data[image][2])
        h = int(bounding_box_data[image][3])

        path = get_image_path(image)

        img = cv2.imread(path)[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(cropped_output, image), img)


def crop_on_annotations():
    """
        creates crop of images using annotations (not model predictions)
    """
    if len(os.listdir(cropped_output)) == 0:
        annotations = load_annotations()
        image_list = create_image_list(annotations)
        crop_images(image_list, annotations)


def get_image_label(image, one_hot=False):
    """
    Args:
        one_hot: specifies if labels should be in one hot vector format ([0,0,0,1,0,0,0])
    """

    for i in range(len(classes)):
        path = os.path.join(workspace, 'train', classes[i], image)
        if os.path.isfile(path):
            if one_hot:
                vec = [0] * len(classes)
                vec[i] = 1
                return vec
            else:
                return classes[i]

    return None


def get_cropped_input_data(size, one_hot=False):
    """
    Args:
        one_hot: specifies if labels should be in one hot vector format ([0,0,0,1,0,0,0])
    Returns:
        list of cropped images and corresponding labels
    """

    image_list = os.listdir(cropped_output)

    X = []
    Y = []
    for image in image_list:

        cropped_path = os.path.join(cropped_output, image)
        img = cv2.imread(cropped_path)

        if img is None:
            continue

        label = get_image_label(image, one_hot)
        if label is not None:
            img = cv2.resize(img, size)
            X.append(img)
            Y.append(label)

    return X, Y


def get_resized_image_path(imagename):
    return os.path.join(resized_output, imagename) + '.npy'


def get_image_path(imagename):
    return os.path.join(image_dir, imagename)


def get_cropped_image_path(imagename):
    return os.path.join(cropped_output, imagename)


if __name__ == "__main__":
    resize_images_and_annotations()
    resize_images_to_npy(384, 384)
    crop_on_annotations()

