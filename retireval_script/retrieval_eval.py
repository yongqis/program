#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import retireval_script.retrieval_utils as tool
from object_detection.utils import label_map_util


def main(pb_model_path, label_path, gt_xml_dir, test_dir, gallery_path, sim_threshold=0.6, top_k=1):
    tool.build_inference_graph(pb_model_path)
    label_map = label_map_util.get_label_map_dict(label_path)

    sess = tf.Session()
    for im in os.listdir(test_dir):
        if not im.endswith('jpg'):
            continue
        im_path = os.path.join(test_dir, im)
        gt_xml_path = os.path.join(gt_xml_dir, '{}.xml'.format(im.split('.')[0]))
        print('正在预测图片', im)
        origin_im = cv.imread(im_path)
        image = cv.cvtColor(origin_im, cv.COLOR_BGR2RGB)
        batch_image = np.expand_dims(image, 0)
        # 得到模型输出数据
        detection_dict = tool.get_output(batch_image, sess)

        # 得到预测框batch_boxes，预测框与库中图片相似度最高的k个图片的相似度值batch_sim_k和对应的batch_label_k
        batch_label_k, batch_sim_k, batch_boxes = \
            tool.batch_retrieval(detection_dict, gallery_path, sim_threshold, top_k)

        # 从xml中读取与gt数据，与预测数据计算后得到precision和recall
        precision, recall = tool.metric(batch_boxes, batch_sim_k, batch_label_k, gt_xml_path, label_map)
        print(precision, recall)

        # 查看预测框
        output_img = origin_im.copy()
        boxes = detection_dict['detection_boxes']
        tool.draw_and_save(output_img, boxes, label_map)
    sess.close()


if __name__ == '__main__':
    pb = r'/home/hnu/workspace/syq/retrieval/data/all_class/saved_model/frozen_inference_graph.pb'
    test = r'/home/hnu/workspace/syq/retrieval/test/query'
    label_map = r'/home/hnu/workspace/syq/retrieval/data/all_class/label_map.pbtxt'
    gallery_path = r'/home/hnu/workspace/syq/retrieval/data/all_class/saved_model/gallery_info.pkl'
    gt_xml = r'/home/hnu/workspace/syq/retrieval/test/xml_updata'
    main(pb, test_dir=test, label_path=label_map, gt_xml_dir=gt_xml, gallery_path=gallery_path)
