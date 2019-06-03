#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
前言
一些数据约定
label_list：元素由label name组成，顺序按照label_map内的顺序， 且第一个位置即label_list[0]为负类
gt_boxes: 标注框组成的np.array，shape为[num_gt, 4] 要不要归一化坐标！！
gt_labels: 标注框label组成的np.array， shape为[num_gt, 1]， 视为行向量计算时记得reshape

'detection_boxes': [1, max_detections, 4]
'detection_scores': [1, max_detections]
'num_detections': [batch]
"""
import cv2 as cv
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import scipy.spatial.distance as distance

from sklearn.externals import joblib
from object_detection_updata.utils import visualization_utils as vis_utils


def build_inference_graph(inference_graph_path):
    """Loads the inference graph and connects it to the input image.
    """
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
        # Imports the graph from `graph_def` into the current default `Graph`-detection_graph
        tf.import_graph_def(graph_def, name='')
    return graph_def


def get_output(input_tensor, sess):
    graph = tf.get_default_graph()
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')
    embeddings = graph.get_tensor_by_name('box_embeddings:0')
    (scores, boxes, embeddings, num_detections) = sess.run(
        [scores, boxes, embeddings, num_detections], feed_dict={image_tensor: input_tensor})

    boxes = np.squeeze(boxes)  # 把shape中为1的维度去掉，主要是batch_size
    embeddings = np.squeeze(embeddings)
    scores = np.squeeze(scores)
    num_detections = np.squeeze(num_detections)

    detections_dict = {'detection_boxes': boxes,
                       'detection_scores': scores,
                       'num_detections': num_detections,
                       'box_embeddings': embeddings,
                       }
    return detections_dict


def batch_retrieval(detection_dict, gallery_path, sim_threshold, top_k):
    """
    把batch个embedding，去数据集中进行检索 取top-k
    设置度量方式，
    相似性阈值，
    返回top-k的类别，数量不足填充0
    """
    # 分解
    boxes = detection_dict['detection_boxes']  # [max_detection, 4]
    embeddings = detection_dict['box_embeddings']  # [max_detection, 128]
    num_detections = detection_dict['num_detections']  #

    # load
    gallery_embeddings, gallery_label_id = joblib.load(gallery_path)  # shape[]

    gallery_embeddings = np.squeeze(np.array(gallery_embeddings))  # [num_gallery, 128]
    gallery_label_id = np.array(gallery_label_id)  # [num_gallery,]

    # compute 余弦相似度，取值范围[-1, 1],1表示相同，
    # 行序是box个数，列序是gallery图片个数
    batch_sim = np.dot(embeddings, gallery_embeddings.T)
    batch_sim = 0.5 + 0.5*batch_sim

    # batch_sim = distance.cdist(embeddings, gallery_embeddings, metric='euclidean')
    # batch_sim = 1/(1+batch_sim)
    # 排序，返回排序后的索引，默认是升序，而需要降序，故取反
    batch_sim_indices = np.argsort(-batch_sim, axis=1)

    # 取top-k的score和对应的label
    batch_sim_indices_k = batch_sim_indices[:, :top_k]

    batch_label_k = np.zeros([num_detections, top_k])
    batch_sim_k = np.zeros([num_detections, top_k])
    batch_boxes = np.zeros([num_detections, 4])
    for i in range(num_detections):
        batch_boxes[i] = boxes[i]
        for j in range(top_k):
            idx = batch_sim_indices_k[i, j]  # 取索引
            # if batch_sim[i, idx] > sim_threshold:
            # print(batch_sim[i, idx])
            batch_label_k[i, j] = gallery_label_id[idx]  # label用数字id代替，最后再转换成文字输出
            batch_sim_k[i, j] = batch_sim[i, idx]
            # else:
            #     break  # 因为使用了排序，当有一个值小于阈值时，后面的score均小于阈值，都设为0

    return batch_label_k, batch_sim_k, batch_boxes


def metric(proposal_boxes, predict_sim_topk, predict_labels_topk, gt_xml, label_map):
    """
    准备数据
    1.推荐框的真正类别array
    2.推荐框的预测类别array
    3.推荐框真正正类的数量
    4.推荐框预测正确的正类的数量 --作为分子
    5.标注框的数量
    precison = 预测正确的正样本数量/推荐框正样本的数量
    recall = 预测正确的正样本数量/所有标注框的数量

    正类类别从1开始，负类即为匹配的推荐框计为0类别

    :param proposal_boxes: np.array [num_detections, 4]
    :param proposal_boxes_scores: np.array [num_detections, 1]
    :param predict_sim_topk: np.array [num_detections, k]
    :param predict_labels_topk: np.array [num_detections, k]
    :param gt_xml: xml path
    :param label_map: read from pbtxt file
    :return: recall p
    """

    gt_boxes, gt_labels = _get_gt(gt_xml, label_map)  # 返回两个np.array()类型gt_label_id from 1 to end
    gt_boxes_num = np.size(gt_labels)

    # 1.将proposal_boxes, gt_boxes, gt_labels传入 target_assign得到proposal_box_target_label
    proposal_box_target_labels = _target_assign(proposal_boxes, gt_boxes, gt_labels, matched_threshold=0.7)
    # 2.根据predict_sim_topk, predict_labels_topk,用多数原则，得到proposal_boxes_pre_label
    proposal_boxes_pre_labels = np.squeeze(predict_labels_topk[:, 0])
    print('box_gt_label: ', proposal_box_target_labels)
    print('box_pd_label', proposal_boxes_pre_labels)
    # 3.对1用大于运算，得到proposal_boxes_gt_label_pos_num
    proposal_boxes_target_label_pos_num = np.sum(np.greater(proposal_box_target_labels, 0))
    # 4.对1和2做逻辑与运算，得到proposal_boxes_pre_label_true_pos_num
    proposal_boxes_pre_label_true_pos_num = \
        np.sum(np.logical_and(np.logical_and(proposal_box_target_labels, proposal_boxes_pre_labels),
                              np.equal(proposal_box_target_labels, proposal_boxes_pre_labels)))
    # 计算准确率precision=4./3.
    precision = proposal_boxes_pre_label_true_pos_num/proposal_boxes_target_label_pos_num
    # 计算召回率recall=4./5.
    recall = proposal_boxes_pre_label_true_pos_num/gt_boxes_num
    # 计算AP
    # 计算mAP
    return precision, recall


def _get_gt(xml_path, label_map):
    # 解析xml obj是list[info], info[label_id, ymin, xmin, ymax, xmax]
    objects = _read_xml_as_eval_info(xml_path, label_map)['objects']
    objects = np.array(objects)  # shape[batch, 5]

    if len(objects) == 0:
        gt_boxes = None
        gt_labels = None
    else:
        # 划分出label_id和boxe坐标
        gt_labels, gt_boxes = np.hsplit(objects, [1])  # [num_gt, 1]
    return gt_boxes, gt_labels


def _read_xml_as_eval_info(xml_path, label_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    info = {}
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    info['shape'] = (height, width)
    objects = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if label_map.get(cls_name, 0) is 0:
            continue
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text) / width
        ymin = int(xml_box.find('ymin').text) / height
        xmax = int(xml_box.find('xmax').text) / width
        ymax = int(xml_box.find('ymax').text) / height
        objects.append([label_map[cls_name], ymin, xmin, ymax, xmax])
    info['objects'] = objects
    return info


def _target_assign(predict_boxes, gt_boxes, gt_labels, matched_threshold):
    """
    :param predict_boxes: [detection_num, 4]
    :param gt_boxes: [gt_num,4]
    :param gt_labels: [gt_num,1]
    :return:
    """
    # 计算得到IOU矩阵 [num_gt_boxes, num_detections]
    y_min_p, x_min_p, y_max_p, x_max_p = np.split(predict_boxes, 4, 1)  # split 确保shape[num,1]
    y_min_t, x_min_t, y_max_t, x_max_t = np.split(gt_boxes, 4, 1)
    # 1.计算intersection [num_detections, num_gt_boxes]
    all_pairs_min_y_max = np.minimum(y_max_t, np.transpose(y_max_p))  # e.g.shape[4,1] shape[1,6] = shape[4,6]
    all_pairs_max_y_min = np.maximum(y_min_t, np.transpose(y_min_p))
    intersection_h = np.maximum(0.0, all_pairs_min_y_max - all_pairs_max_y_min)

    all_pairs_min_x_max = np.minimum(x_max_t, np.transpose(x_max_p))
    all_pairs_max_x_min = np.maximum(x_min_t, np.transpose(x_min_p))
    intersection_w = np.maximum(0.0, all_pairs_min_x_max - all_pairs_max_x_min)

    intersection_areas = intersection_h * intersection_w
    # 2.计算每组每个boxes的面积 自动使用广播扩展维度
    print(type((y_max_p - y_min_p) * (x_max_p - x_min_p)))
    areas_p = np.squeeze((y_max_p - y_min_p) * (x_max_p - x_min_p), 1)
    areas_t = np.squeeze((y_max_t - y_min_t) * (x_max_t - x_min_t), 1)
    # 3.unions = (box1+box2-intersection)
    unions = np.expand_dims(areas_p, 0) + np.expand_dims(areas_t, 1) - intersection_areas
    # 4.np.where(condition, x, y) if condition为true, 返回x对应位置的值， else 返回y对应位置的值
    similarity_matrix = np.where(np.equal(intersection_areas, 0.0),
                                 np.zeros_like(intersection_areas),
                                 np.true_divide(intersection_areas, unions))  # [gt_num, pre_num]
    # 根据IOU矩阵确定每个predict_boxe的对应的gt_box索引
    def _match_when_rows_are_empty():
        """Performs matching when the rows of similarity matrix are empty.

        When the rows are empty, all detections are false positives. So we return
        a tensor of -1's to indicate that the columns do not match to any rows.

        Returns:
          matches:  int32 tensor indicating the row each column matches to.
        """
        return -1 * np.ones([np.shape(similarity_matrix)[1]], dtype=np.int32)

    def _match_when_rows_are_non_empty():
        # Matches for each column
        # 1.得到每列（每个预测框）的IOU最大值所在行（标注框）
        matches = np.argmax(similarity_matrix, axis=0)  # shape[num_detections,]

        # 2.得到每列的最大值
        matched_vals = np.max(similarity_matrix, 0)  # shape[num_detections,]

        # 3.按照阈值要求划分正负类，并修改matches中对应位置的值为-1
        indicator = np.greater(matched_threshold, matched_vals)  # 低于阈值的位置，
        matches = np.add(np.multiply(matches, 1 - indicator), -1 * indicator)  # 先将负类对应位置变为0，再-1

        return matches

    if np.greater(np.shape(similarity_matrix)[0], 0):
        target_box_id = _match_when_rows_are_non_empty()
    else:
        target_box_id = _match_when_rows_are_empty()

    # 确定target_label //target_box_id shape[num_detections,]
    # 1.得到正负类box在match结果中的位置 即正负类预测框在序列中的位置
    matched_box_indices = np.reshape(np.where(np.greater(target_box_id, -1)), [-1])
    unmatched_box_indices = np.reshape(np.where(np.greater(0, target_box_id)), [-1])
    # 2.1得到正类box对应的第几个标注框，即标注框的索引
    matched_gt_indices = _gather(target_box_id, matched_box_indices)
    # 2.2得到正类box对应标注框的类别,shape为[num_matched,d_1]
    matched_label_targets = _gather(np.squeeze(gt_labels), matched_gt_indices)
    # list乘法的意义，3*[1]=[1,1,1]  list加法的意义[tf.size]+[1,1]=[tf.size,1,1]
    # 3.1得到负类box对应的类别 shape[num_unmatched,1]
    # unmatched_label_targets = np.array([1] + num_classes * [0], dtype=np.float32)  # 负类的one-hot编码
    unmatched_label_targets = np.array([0], dtype=np.float32)  # 负类序列编码
    unmatched_label_targets = np.tile(unmatched_label_targets, [np.size(unmatched_box_indices)])
    # 4.还原
    matched_label_targets = np.squeeze(matched_label_targets)
    unmatched_label_targets = np.squeeze(unmatched_label_targets)

    targets_labels = _dynamic_stitch([matched_box_indices, unmatched_box_indices],
                                     [matched_label_targets, unmatched_label_targets],
                                     predict_boxes.shape[0])

    return targets_labels


def _gather(x, indices):
    read = [x[i] for i in indices]
    return np.array(read)


def _dynamic_stitch(indices_list, data_list, data_num):
    res = []
    while len(indices_list):
        num = len(indices_list)
        # 每次比较每个array的开头，保存最小值
        cur_min = data_num
        row = 0
        for i in range(num):
            if indices_list[i][0] < cur_min:
                cur_min = indices_list[i][0]
                row = i
        # 保存并删除
        res.append(data_list[row][0])

        data_list[row] = np.delete(data_list[row], 0)
        indices_list[row] = np.delete(indices_list[row], 0)

        if np.size(data_list[row]) == 0:
            del data_list[row]
            del indices_list[row]
    return np.array(res)


def draw_and_save(output_img, boxes, label_map):
    """
    将预测区域和类别画在图片上保存
    :return:
    """

    output_img = vis_utils.visualize_boxes_and_labels_on_image_array(
        output_img, boxes, None, None, label_map,
        use_normalized_coordinates=True,
        min_score_thresh=0.0,
        line_thickness=3,
        agnostic_mode=True)
    cv.imshow('hei', output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
