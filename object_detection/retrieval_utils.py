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
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
import xml.etree.ElementTree as ET


def build_inference_graph(inference_graph_path):
    """Loads the inference graph and connects it to the input image.
    """
    with tf.gfile.Open(inference_graph_path, 'r') as graph_def_file:
        graph_content = graph_def_file.read()
    graph_def = tf.GraphDef()
    graph_def.MergeFromString(graph_content)
    tf.import_graph_def(graph_def, name='')


def get_output(input_tensor):
    graph = tf.get_default_graph()
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    boxes = graph.get_tensor_by_name('detection_boxes:0')
    num_detections = graph.get_tensor_by_name('num_proposals:0')
    embeddings = graph.get_tensor_by_name('obj_embeddings:0')

    (boxes, embeddings, num_detections) = tf.get_default_session().run(
        [boxes, embeddings, num_detections],
        feed_dict={image_tensor: input_tensor})

    boxes = np.squeeze(boxes)  # 把shape中为1的维度去掉，主要是batch_size
    embeddings = np.squeeze(embeddings)

    detections_dict = {'norm_proposal_boxes': boxes,
                       'num_proposals': num_detections,
                       'obj_embeddings': embeddings}
    return detections_dict


def batch_retrieval(detection_dict, gallery_dir, sim_threshold, top_k):
    """
    把batch个embedding，去数据集中进行检索 取top-k
    设置度量方式，
    相似性阈值，
    返回top-k的类别，数量不足填充0
    """
    boxes = detection_dict['norm_proposal_boxes']  # [max_detection, 4]
    scores = detection_dict['proposal_box_scores']  # [max_detection, ]
    embeddings = detection_dict['obj_embeddings']  # [max_detection, 128]
    num_detections = detection_dict['num_proposals']  # [1]
    # load
    gallery_embeddings, gallery_label = gallery_dir

    # compute similar batch_score 行序是box个数，列序是gallery图片个数
    batch_sim = np.dot(embeddings, gallery_embeddings.T)
    # 排序，返回排序后的索引
    batch_sim_indices = np.argsort(-batch_sim)
    # 取top-k的score和对应的label
    batch_sim_indices_k = batch_sim_indices[:, :top_k]

    batch_label_k = np.zeros([num_detections, top_k])
    batch_sim_k = np.zeros([num_detections, top_k])
    batch_boxes = np.zeros([num_detections, 4])
    batch_scores = np.zeros([num_detections, 1])  # 正类置信度
    for i in range(num_detections):
        batch_boxes[i] = boxes[i]
        batch_scores[i] = scores[i]
        for j in range(top_k):
            idx = batch_sim_indices_k[i, j]
            if batch_sim[i, idx] > sim_threshold:
                batch_label_k[i, j] = gallery_label[idx]  # label用数字id代替，最后再转换成文字输出
                batch_sim_k[i, j] = batch_sim[i, idx]
            else:
                break  # 因为使用了排序，当有一个值小于阈值时，后面的score均小于阈值，都设为0

    return batch_label_k, batch_sim_k, batch_boxes, batch_scores


def metric(proposal_boxes, proposal_boxes_scores, predict_sim_topk, predict_labels_topk, gt_xml, label_path):
    """
    准备数据
    1.推荐框的真正类别array
    2.推荐框的预测类别array
    3.推荐框真正正类的数量
    4.推荐框预测正确的正类的数量 --作为分子
    5.标注框的数量
    precison = 预测正确的正样本数量/推荐框正样本的数量
    recall = 预测正确的正样本数量/所有标注框的数量
    :param proposal_boxes: np.array [num_detections, 4]
    :param proposal_boxes_scores: np.array [num_detections, 1]
    :param predict_sim_topk: np.array [num_detections, k]
    :param predict_labels_topk: np.array [num_detections, k]
    :param gt_xml: xml path
    :param label_list: all label name include neg as 0 index in this list are sorted
    :return: recall p
    """
    label_map = label_map_util.load_labelmap(label_path)
    label_list = label_map_util.convert_label_map_to_categories(label_map)
    gt_boxes, gt_labels = _get_gt(gt_xml, label_list)  # 返回两个np.array()类型
    gt_boxes_num = np.size(gt_labels)

    # 1.将proposal_boxes, gt_boxes, gt_labels传入 target_assign得到proposal_box_target_label
    proposal_box_target_labels = _target_assign(proposal_boxes, gt_boxes, gt_labels, matched_threshold=0.7)
    # 2.根据predict_sim_topk, predict_labels_topk,用多数原则，得到proposal_boxes_pre_label
    proposal_boxes_pre_labels = predict_labels_topk
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


def _get_gt(xml_path, label_list):
    # 解析xml obj是list[info], info[label_id, ymin, xmin, ymax, xmax]
    objects = _read_xml_as_eval_info(xml_path, label_list)['objects']
    objects = np.array(objects)  # shape[batch, 5]
    print(objects.shape)
    if len(objects) == 0:
        gt_boxes = None
        gt_labels = None
    else:
        # 划分出label_id和boxe坐标
        gt_labels, gt_boxes = np.hsplit(objects, [1])
        print(gt_labels.shape)
    return gt_boxes, gt_labels


def _read_xml_as_eval_info(xml_path, label_list):
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
        if cls_name not in label_list:
            continue
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text) / width
        ymin = int(xml_box.find('ymin').text) / height
        xmax = int(xml_box.find('xmax').text) / width
        ymax = int(xml_box.find('ymax').text) / height
        objects.append([label_list.index(cls_name), ymin, xmin, ymax, xmax])
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
    areas_p = np.squeeze((y_max_p - y_min_p) * (x_max_p - x_min_p), [1])
    areas_t = np.squeeze((y_max_t - y_min_t) * (x_max_t - x_min_t), [1])
    # 3.unions = (box1+box2-intersection)
    unions = areas_p + areas_t - intersection_areas
    # 4.np.where(condition, x, y) if condition为true, 返回x对应位置的值， else 返回y对应位置的值
    similarity_matrix = np.where(np.equal(intersection_areas, 0.0),
                                 np.zeros_like(intersection_areas),
                                 np.true_divide(intersection_areas, unions))

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
        """Performs matching when the rows of similarity matrix are non empty.

        Returns:
          matches:  int32 np.array indicating the row each column matches to.
        """
        # Matches for each column
        # 1.得到每列（每个预测框）的IOU最大值所在行（标注框）。即每个anchor Iou最大的那个标注框
        matches = np.argmax(similarity_matrix, axis=0)  # shape[num_detections,]
        # 2.得到每列的最大值
        matched_vals = np.max(similarity_matrix, 0)  # shape[num_detections,]
        # 3.按照阈值要求找出predict_box中的负类索引，并修改matches对应位置的值为-1，表示该box的位置不准确，视为负类
        below_matched_threshold = np.greater(matched_threshold, matched_vals)  # bool shape[num_detections, ]
        indicator = np.cast(below_matched_threshold, np.bool)
        matches = np.add(np.multiply(matches, 1 - indicator), -1 * indicator)  # 先将负类对应位置变为0，再-1
        return np.cast(matches, np.int32)

    if np.greater(np.shape(similarity_matrix)[0], 0):
        target_box_id = _match_when_rows_are_non_empty()
    else:
        target_box_id = _match_when_rows_are_empty()

    # 确定target_label //target_box_id shape[num_detections,]
    # 1.得到正负类box在match结果中的位置
    matched_box_indices = np.cast(np.reshape(np.where(np.greater(target_box_id, -1)), [-1]), np.int32)
    unmatched_box_indices = np.cast(np.reshape(np.where(np.greater(0, target_box_id)), [-1]), np.int32)

    # 2.1得到正类box对应的第几个标注框，即标注框的索引
    matched_gt_indices = _gather(target_box_id, matched_box_indices)
    matched_gt_indices = np.cast(np.reshape(matched_gt_indices, [-1]), np.int32)
    # 2.2得到正类box对应标注框的类别,shape为[num_matched,d_1]
    matched_label_targets = _gather(gt_labels, matched_gt_indices)  # [match_num, 1]

    # list乘法的意义，3*[1]=[1,1,1]  list加法的意义[tf.size]+[1,1]=[tf.size,1,1]
    # 3.1得到负类box对应的类别 shape[num_unmatched,1]
    # unmatched_label_targets = np.array([1] + num_classes * [0], dtype=np.float32)  # 负类的one-hot编码
    unmatched_label_targets = np.array([-1], dtype=np.float32)  # 负类序列编码
    ones = 1*[1]
    # np.stack([array1, array2...]) =>
    unmatched_label_targets = np.tile(np.expand_dims(unmatched_label_targets, 0),
                                      np.stack([np.size(unmatched_box_indices)] + ones))  # [unmatch_num, 1]
    # 4.还原
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


def draw_and_save(self):
    """
    将预测区域和类别画在图片上保存
    :return:
    """
    pass
