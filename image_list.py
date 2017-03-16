import json
import cv2
import numpy as np
import os
import tflearn as tf
from tensorflow.python.platform import gfile

CURRENT = os.path.dirname(__file__)
BOX_ANNOTATIONS_PATH = os.path.join(CURRENT, 'annotations', 'all.json')

def create_annotations():
    """
    Returns:
        dictionary containing annotation box for every filename
    """
    with open(BOX_ANNOTATIONS_PATH) as data_file:
        bounding_box_data = json.load(data_file)

    annotations = {}
    for i in range(len(bounding_box_data)):
        filename = bounding_box_data[i]['filename']

        xstart = bounding_box_data[i]['annotations'][0]['x']
        xend = bounding_box_data[i]['annotations'][0]['width']
        ystart = bounding_box_data[i]['annotations'][0]['y']
        yend = bounding_box_data[i]['annotations'][0]['height']
        annotations[filename] = [xstart, xend, ystart, yend]

    return annotations

def create_image_list(image_dir, annotations):
  extension = 'jpg'
  image_list = []
  file_glob = os.path.join(image_dir, '*.' + extension)
  file_list = gfile.Glob(file_glob)

  for file_name in file_list:
      if os.path.basename(file_name) in annotations.keys():
          image_list.append(file_name)
  return image_list


