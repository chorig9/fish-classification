from tensorflow.python.platform import gfile
import os.path
import numpy as np
from image_list import *
from cv2 import *

def get_train_data_for_localization():
    # Get workspace address
    workspace = os.path.dirname(__file__)
    
    # Get path to directory which contains images
    images_path = os.path.join(workspace, 'train', 'all')

    # Retrieve annotations
    annotations = create_annotations()

    # Get all filepaths
    filepaths = create_image_list(images_path, annotations)

    image_data = []
    bounding_box_data = []

    # Create X and Y
    counter = 1
    for image_path in filepaths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1280, 720))

        image_data.append(image)
        counter = counter + 1

        filename = os.path.basename(image_path)
        bounding_box_data.append(annotations[filename])

        if counter %100 == 0:
            # Return X and Y for regression
            print("Wyslalem paczke")
            yield image_data, bounding_box_data
            image_data = []
            bounding_box_data = []

    return image_data, bounding_box_data

get_train_data_for_localization()