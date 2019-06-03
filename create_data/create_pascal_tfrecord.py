"""
step1 图片信息整理
step2 创建一个tfrecord文件/类型
step3 **将一个样本的信息写入字典dict
step4 **将字典转换成tf.train.Feature类型
step5 根据tf.train.Feature将此样本转成tf.train.Example类型 tf.train.Example(feature=tf.train.Feature(feature={}))
step6 将tf.train.Example序列化
step7 将序列化后的样本写入tfrecord
"""
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import time
import logging
import hashlib
import tensorflow as tf
import xml.etree.ElementTree as ET

from object_detection_updata.utils import dataset_util
from object_detection_updata.utils import label_map_util


flags = tf.flags
# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
flags.DEFINE_string('data_dir', r'D:\Picture\Nestle\Nestle_all', 'Root directory to raw dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('xml_dir', r'D:\Picture\Nestle\Nestle_all\all_xml\xml_updata', 'xml files have the same filename')
flags.DEFINE_string('output_path', r'D:\Picture\Nestle\Nestle_all', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', r'D:\Picture\Nestle\Nestle_all\label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')

# FLAGS是一个dict
FLAGS = flags.FLAGS
SETS = ['train', 'val']


def dict_to_tf_example(xml_path, img_path, label_map_dict):

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    key = hashlib.sha256(encoded_jpg).hexdigest()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    # 需要录入的信息
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []
    difficult_obj = []
    truncated = []
    poses = []

    # 归一化图片里label的位置坐标
    for obj in root.iter('object'):
        if obj.find('name').text not in label_map_dict:
            continue
        difficult = bool(int(obj.find('difficult').text))  # difficult value is 0 or 1
        xml_box = obj.find('bndbox')
        _xmin = float(xml_box.find('xmin').text) / width
        _xmax = float(xml_box.find('xmax').text) / width
        _ymin = float(xml_box.find('ymin').text) / height
        _ymax = float(xml_box.find('ymax').text) / height

        if (0 <= _xmin < _xmax <= 1) and (0 <= _ymin < _ymax <= 1):
            xmin.append(_xmin)
            xmax.append(_xmax)
            ymin.append(_ymin)
            ymax.append(_ymax)
            difficult_obj.append(int(difficult))
            classes_text.append(obj.find('name').text.encode('utf8'))
            classes.append(label_map_dict[obj.find('name').text])

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(os.path.basename(xml_path).encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(os.path.basename(img_path).encode('utf8')),
            'image/key/sha256':  dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    # Reads a label_map.pbtxt and returns a dictionary of label names to id.
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    path_set = os.path.join(FLAGS.data_dir, FLAGS.set)
    xml_dir = FLAGS.xml_dir
    record_path = os.path.join(FLAGS.output_path, '{}.record'.format(FLAGS.set))

    print("Please check image type in follow set: ")
    file_type = set()
    for fileName in os.listdir(path_set):
        file_type.add(fileName.split('.')[-1])
    print(file_type)
    time.sleep(5)
    print("start write to record files")
    # A class to write records to a TFRecords file.
    writer = tf.python_io.TFRecordWriter(record_path)

    for fileName in os.listdir(path_set):
        file_name = fileName.split('.')[0]
        if fileName.endswith('jpg') or fileName.endswith('jpeg'):
            pic_path = os.path.join(path_set, fileName)  # 图片路径
            try:
                # print('writing info from: {}'.format(pic_path))
                xml_path = os.path.join(xml_dir, '{}.xml'.format(file_name))  # 对应xml路径
                example = dict_to_tf_example(xml_path=xml_path, img_path=pic_path, label_map_dict=label_map_dict)
                writer.write(example.SerializeToString())  # 转化为字符串

            except IOError:
                print('WARNING--error path: {}'.format(pic_path))
                continue

    writer.close()
    print('finished write')


if __name__ == '__main__':
    tf.app.run()

