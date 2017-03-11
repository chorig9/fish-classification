import numpy as np
import tensorflow as tf
import os

from PIL import Image
from matplotlib import patches
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

home = os.path.expanduser('~')
imageDir = os.path.join(home, 'fish/test_stg1')
modelFullPath = os.path.join(home, 'fish/output_graph.pb')


def create_image_lists(image_dir):
  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
  file_list = []
  for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(gfile.Glob(file_glob))
  return file_list


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        for imagePath in sorted(create_image_lists(imageDir)):
            image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
            boxes_values = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(boxes_values,
                                   {'DecodeJpeg/contents:0': image_data})

            im = np.array(Image.open(imagePath), dtype=np.uint8)
            ax.imshow(im)

            predictions = predictions[0]
            x = predictions[0]
            y = predictions[1]
            width = predictions[1] - predictions[0]
            height = predictions[3] - predictions[2]

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            plt.pause(0.5)
            plt.cla()


if __name__ == '__main__':
    run_inference_on_image()




