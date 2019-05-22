#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from sklearn.externals import joblib
from object_detection.utils import label_map_util


def main(pb_model_path, gallery_dir, label_map_path, save_path):
    label_map = label_map_util.get_label_map_dict(label_map_path)
    gallery_embeddings = []
    gallery_label_id = []
    # 加载模型
    g_def = tf.GraphDef()
    with tf.gfile.GFile(pb_model_path, 'rb') as fid:
        g_def.ParseFromString(fid.read())
    tf.import_graph_def(g_def, name='')

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    embeddings_tensor = tf.get_default_graph().get_tensor_by_name('embeddings:0')
    sess = tf.Session()
    count = 1
    for cur_fold, sub_folds, files in os.walk(gallery_dir):
        # for sub_fold in sub_folds:
        #     res = label_map.get(sub_fold, 55)
        #     if res is 55:
        #         print(sub_fold)
        for file in files:
            if not file.endswith('jpg'):
                continue
            im = cv.imread(os.path.join(cur_fold, file))
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            if im.shape[0] < 33 or im.shape[1] < 33:
                # print(os.path.join(cur_fold, file))
                continue
            im = np.expand_dims(im, 0)
            # 送入模型,得到结果
            embeddings = sess.run(embeddings_tensor, feed_dict={image_tensor: im})
            print(count)
            count += 1
            print(cur_fold)
            gallery_embeddings.append([embeddings[0]])
            # label转换 string=>num
            label_name = os.path.relpath(cur_fold, gallery_dir)
            gallery_label_id.append(label_map[label_name])
    sess.close()
    joblib.dump([gallery_embeddings, gallery_label_id], save_path)


if __name__ == '__main__':
    pb_model_path = r'/home/hnu/workspace/syq/retrieval/data/all_class/saved_model/frozen_build_gallery_graph.pb'
    gallery_dir = r'/home/hnu/workspace/syq/retrieval/test/gallery'
    label_map_path = r'/home/hnu/workspace/syq/retrieval/data/all_class/label_map.pbtxt'
    pkl_save_path = r'/home/hnu/workspace/syq/retrieval/data/all_class/saved_model/gallery_info.pkl'
    main(pb_model_path, gallery_dir, label_map_path, pkl_save_path)
