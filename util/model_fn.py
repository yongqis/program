import functools
import numpy as np
import tensorflow as tf
from util.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss
from util.update_util import clip_gradient_norms
from object_detection_updata.utils import config_util
from object_detection_updata.utils import ops as util_ops
from object_detection_updata.utils import variables_helper
from object_detection_updata.builders import model_builder
from object_detection_updata.builders import optimizer_builder


def model_fn(images, labels, mode, params):
    """

    :param images: images是一个[batch, height, width, 3] np.ndarray
    :param labels: labels需进一步分解 labels=[batch_box, batch_label]=[np.array/[1, num_box, 4], np.array/[1,num_box]]
    :param mode:
    :param params:
    :return:
    """
    configs = config_util.get_configs_from_pipeline_file(params)
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        detection_model = model_builder.build(model_config=model_config, is_training=is_training)  # 初始化模型的辅助工具
        # label data preprocess
        num_classes = detection_model.num_classes
        batch_boxes, batch_classes = labels
        classes = tf.cast(batch_classes[0], tf.int32)
        classes -= 1
        classes_gt = util_ops.padded_one_hot_encoding(indices=classes, depth=num_classes, left_pad=0)
        boxes_gt = batch_boxes[0]
        groundtruth_boxes_list, groundtruth_classes_list = [boxes_gt], [classes_gt]

        detection_model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list)

        # 数据类型处理
        # batch维度的大小固定下来, resize和normalize Tensor.get_shape.as_list只能读取静态shape
        image = images[0]
        batch_images = tf.expand_dims(image, 0)
        batch_images = tf.cast(batch_images, tf.float32)

        # 开始送入模型
        batch_images = detection_model.preprocess(batch_images)  # 预处理
        rpn_prediction_dict = detection_model.predict_rpn(batch_images)  # 第一阶段预测结果
        rpn_detection_dict = detection_model.postprocess_rpn(rpn_prediction_dict)  # 第一阶段预测结果后处理
        rpn_loss_dict = detection_model.loss_rpn(rpn_prediction_dict)  # 第一阶段loss
        for value in rpn_loss_dict.values():
            tf.losses.add_loss(value)
        if model_config.faster_rcnn.first_stage_only:
            final_prediction_dict = detection_model.predict_second_stage(rpn_detection_dict)  # 第二阶段预测结果
            final_detection_dict = detection_model.postprocess_box_classifier(final_prediction_dict)
            final_loss_dict = detection_model.loss_box_classifier(final_prediction_dict)
            for value in final_loss_dict.values():
                tf.losses.add_loss(value)
    # 汇总loss
    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')
    # 全局步数
    global_step = tf.train.get_or_create_global_step()
    # 梯度下降的三个结构-学习率衰减吗？梯度累积吗？权重平滑吗？
    # 定义优化器-1 集成
    optimizer = optimizer_builder.build(train_config.optimizer, global_summaries)
    # 定义优化器-2 分步
    learning_rate = tf.train.exponential_decay(params.base_learning_rate,
                                               global_step=global_step,
                                               decay_steps=1000,  # 最好为一个epoch=num_example/batch_size
                                               decay_rate=0.99,
                                               staircase=False)  # staircase是否为阶梯形，即对decay_rate的系数进行向下取整
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # 计算权重（变量）的梯度-组成一个pairs, 所有pairs组成一个list返回
    grads_and_vars = optimizer.compute_gradients(total_loss)
    # 梯度更新前3个约束条件
    # 偏置项参数乘以一个系数
    if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)
    # 冻结指定参数的梯度--筛选掉不需要进行梯度下降的参数
    if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(grads_and_vars, train_config.freeze_variables)
    # 梯度裁剪，防止梯度爆炸--是控制梯度在最大范式范围
    if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
            grads_and_vars = clip_gradient_norms(grads_and_vars, train_config.gradient_clipping_by_norm)

    # 权重（变量）更新
    grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # 更新操作汇总--如BN/moving mean需要单独的更新操作
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)  # tf.group() 星号表达式
    # 指定依赖关系--先执行update_op节点的操作，再执行train_tensor操作
    with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')  # tf.identity()

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_tensor)

