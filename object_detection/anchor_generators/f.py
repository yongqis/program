#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator


def construct_anchor_grid():
    base_anchor_size = [10, 10]
    anchor_stride = [19, 19]
    anchor_offset = [0, 0]
    scales = [0.5, 1.0, 2.0]
    aspect_ratios = [1.0, 1.5]

    exp_anchor_corners = [[-2.5, -2.5, 2.5, 2.5], [-5., -5., 5., 5.],
                          [-10., -10., 10., 10.], [-2.5, 16.5, 2.5, 21.5],
                          [-5., 14., 5, 24], [-10., 9., 10, 29],
                          [16.5, -2.5, 21.5, 2.5], [14., -5., 24, 5],
                          [9., -10., 29, 10], [16.5, 16.5, 21.5, 21.5],
                          [14., 14., 24, 24], [9., 9., 29, 29]]

    anchor_generator = grid_anchor_generator.GridAnchorGenerator(
        scales,
        aspect_ratios,
        base_anchor_size=base_anchor_size,
        anchor_stride=anchor_stride,
        anchor_offset=anchor_offset)

    anchors = anchor_generator.generate(feature_map_shape_list=[(5, 5)])
    print(anchors.get())
    return anchors.get_center_coordinates_and_sizes()
    # anchor_corners = anchors.get()


if __name__ == '__main__':
    with tf.Session() as sess:
        an = construct_anchor_grid()
        print(sess.run(an))
