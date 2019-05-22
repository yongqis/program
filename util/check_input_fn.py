import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.input_fn import train_input_fn

file_path = '/home/hnu/workspace/syq/retrieval/data/train.record'
batch_image, batch_gt_label = train_input_fn(file_path, None)
count = 1
sess = tf.Session()

while True:
    try:
        i, j = sess.run([batch_image, batch_gt_label])
        batch_boxes_array, batch_classes_array = j
        one_hot_array = np.zeros([batch_classes_array.shape[1], 456])
        for i, j in enumerate(batch_classes_array[0]):
            one_hot_array[i, j] = 1
        groundtruth_boxes_list, groundtruth_classes_list = list(batch_boxes_array[0]), list(one_hot_array)
        print(type(groundtruth_classes_list))
        print(type(groundtruth_boxes_list))
        print(groundtruth_classes_list)
        print(count)
        count += 1
        # plt.imshow(np.squeeze(i))
        # plt.show()
        # print(type(j))
        # batch_bounding_box, batch_box_label = j[0], j[1]
        # print(batch_bounding_box.shape)
        # print(batch_box_label.shape)

    except tf.errors.OutOfRangeError:
        break

