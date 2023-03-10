import os
import io
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import parse_frame, int64_feature, bytes_feature, float_list_feature, bytes_list_feature, int64_list_feature
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
}

def process_tfr(path):
    """
    process tf api tf record
    """
    
    print(f'Processing {path}')
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}
    f, ax = plt.subplots(4, 5, figsize=(20, 10))
    for idx, data in enumerate(dataset):
        if idx >= 4*5:
            break
        
        x = idx % 4
        y = idx % 5
        # example = tf.train.Example()
        # example.ParseFromString(data.numpy())
        p_ex = tf.io.parse_single_example(data.numpy(), feature_description)
        # example = tf.train.Example.FromString(data) no
        encoded_jpg_io = io.BytesIO(p_ex['image/encoded'].numpy())
        image = Image.open(encoded_jpg_io)
        ax[x, y].imshow(image)
        
        width = p_ex['image/width'].numpy()
        height = p_ex['image/height'].numpy()
        xmins = (p_ex['image/object/bbox/xmin'].numpy() * width).astype(int)
        ymins = (p_ex['image/object/bbox/ymin'].numpy() * height).astype(int)
        xmaxs = (p_ex['image/object/bbox/xmax'].numpy() * width).astype(int)
        ymaxs = (p_ex['image/object/bbox/ymax'].numpy() * height).astype(int)
        labels = p_ex['image/object/class/label'].numpy()
        
        for x1, y1, x2, y2, l in zip(xmins, ymins, xmaxs, ymaxs, labels):
            rec = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", edgecolor=colormap[l])
            ax[x, y].add_patch(rec)
            
    plt.show()



if __name__ == '__main__':
    process_tfr('output/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord')
    print('Done')