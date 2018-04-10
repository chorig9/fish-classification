import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import data
import re

modelFullPath = "D:\\tmp\\imagenet\\classify_image_graph_def.pb"


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self):
        label_lookup_path = "D:\\tmp\\imagenet\\imagenet_2012_challenge_label_map_proto.pbtxt"
        uid_lookup_path = "D:\\tmp\\imagenet\\imagenet_synset_to_human_label_map.txt"
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
        print(self.node_lookup)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_sliding_window():
    annotations = data.load_annotations()
    filepaths = data.create_image_list(annotations)

    input_vectors, labels = data.get_resized_input_data(filepaths, annotations)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    window_size_x = 200
    window_size_y = 100

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
    create_graph()

    with tf.Session() as sess:

        for imagename, input in zip(filepaths[:5], input_vectors[:5]):
            image = cv2.imread(data.get_image_path(imagename))
            ax1.imshow(image)
            ax1.imshow(image)
            height, width, _ = image.shape

            window_x = 0
            window_y = 0

            positions = []

            while window_x + window_size_x < width and window_y + window_size_y < height:
                sub_image = image[window_y:(window_y + window_size_y), window_x:(window_x + window_size_x)]
                softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
                predictions = sess.run(softmax_tensor,
                                       {'Cast:0': sub_image})
                predictions = np.squeeze(predictions)
                top_k = predictions.argsort()[-1:][::-1]

                fish = False

                for node_id in top_k:
                    human_string = node_lookup.id_to_string(node_id)
                    if 'shark' in human_string or 'fish' in human_string or 'whale' in human_string or 'tuna' in human_string:
                        fish = True

                if fish:
                    rect = patches.Rectangle((window_x, window_y), window_size_x, window_size_y, linewidth=1, edgecolor='r',
                                      facecolor='none')
                    ax1.add_patch(rect)
                    plt.pause(0.1)

                window_x += window_size_x
                if window_x + window_size_x > width:
                    window_x = 0
                    window_y += window_size_y

            plt.pause(0.5)

            plt.cla()
        plt.pause(5)


run_sliding_window()
