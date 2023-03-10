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

def create_tf_example(filename, encoded_jpeg, annotations):
    """
    convert to tensorflow object detection API format
    args:
    - filename [str]: name of the image
    - encoded_jpeg [bytes-likes]: encoded image
    - annotations [list]: bboxes and classes
    returns:
    - tf_example [tf.Example]
    """
    # TO BE IMPLEMENTED
    encoded_jpg_io = io.BytesIO(encoded_jpeg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    filename = filename.encode('utf8')
    image_format = b'jpg'    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    mapping = {0: 'unknown', 1: 'vehicle', 2: 'pedestrian', 3: 'sign', 4: 'cyclist'}
    
    for ann in annotations:
        xmin, ymin = ann.box.center_x - 0.5 * ann.box.length, ann.box.center_y - 0.5 * ann.box.width
        xmax, ymax = ann.box.center_x + 0.5 * ann.box.length, ann.box.center_y + 0.5 * ann.box.width
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)    
        classes.append(ann.type)
        classes_text.append(mapping[ann.type].encode('utf8'))
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    
    return tf_example

def process_tfr(path):
    """
    process a waymo tf record into a tf api tf record
    """
    # create processed data dir
    file_name = os.path.basename(path)

    print(f'Processing {path}')
    output_name = f'output/{file_name}'
    print(f'Output: {output_name}')
    writer = tf.python_io.TFRecordWriter(output_name)
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        encoded_jpeg, annotations = parse_frame(frame)
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
    print(f"Records written: {idx}")


if __name__ == '__main__':
    process_tfr('input/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord')
    print('Done')