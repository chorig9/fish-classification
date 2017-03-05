import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile

home = os.path.expanduser('~')
imageDir = os.path.join(home, 'fish/test_stg1')
modelFullPath = os.path.join(home, 'fish/output_graph.pb')
labelsFullPath = os.path.join(home, 'fish/output_labels.txt')


def create_image_lists(image_dir):
  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
  file_list = []
  for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(gfile.Glob(file_glob))
 #images = []
 #for file_name in file_list:
 #  base_name = os.path.basename(file_name)
 #  images.append(base_name)
  return file_list


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = []
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]

    file = open(os.path.join(home, 'fish/output.csv'), 'w')
    header = "image"
    for label in sorted(labels):
        header += "," + label.upper()
    file.write(header + "\n")

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        for imagePath in sorted(create_image_lists(imageDir)):
            image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-8:][::-1]  # Getting top 8 predictions
            human_string = []
            score = []
            for node_id in top_k:
                human_string.append(labels[node_id])
                score.append(predictions[node_id])

            scores = [x for (y,x) in sorted(zip(human_string, score))]

            file.write(os.path.basename(imagePath) + ',')
            line = ""
            for score in scores:
                line += str(score) + ','
            line = line[:-1]        # drop last coma
            file.write(line + "\n")

    file.close()

if __name__ == '__main__':
    run_inference_on_image()




