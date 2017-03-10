import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
with open('annotations/alb_labels.json') as data_file:
    bounding_box_data = json.load(data_file)

for i in range(len(bounding_box_data)):
    filename = bounding_box_data[i]['filename']
    im = np.array(Image.open('train/ALB/' + filename), dtype=np.uint8)
    ax.imshow(im)

    xstart = bounding_box_data[i]['annotations'][0]['x']
    xend = bounding_box_data[i]['annotations'][0]['width']
    ystart = bounding_box_data[i]['annotations'][0]['y']
    yend = bounding_box_data[i]['annotations'][0]['height']

    rect = patches.Rectangle((xstart,ystart),xend,yend,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    plt.pause(0.5)
    plt.cla()
