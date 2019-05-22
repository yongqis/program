#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import cv2 as cv
import os
import retrieval_utils as tool


def main(pb_model_path, test_dir, embedding_path, sim_threshold, top_k, lable_path):
    tool.build_inference_graph(pb_model_path)
    with tf.Session() as sess:
        for im in os.listdir(test_dir):
            im_path = os.path.join(test_dir, im)
            origin_im = cv.imread(im_path)
            image = cv.cvtColor(origin_im, cv.COLOR_BGR2RGB)
            detection_dict = tool.get_output(image)

            boxes = detection_dict['norm_proposal_boxes']  # [max_detection, 4]
            embeddings = detection_dict['obj_embeddings']  # [max_detection, 128]
            num_detections = detection_dict['num_proposals']  # [1]
            print(boxes.shape)
            print(embeddings.shape)
            print(num_detections.shape)
            # tool.batch_retrieval(output_dict, embedding_path, sim_threshold, top_k)