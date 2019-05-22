# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools
import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from slim.deployment import model_deploy

slim = tf.contrib.slim


# 文件处理队列
# 将 文件读取队列的结果 create_tensor_dict_fn 里的数据，进行图片等信息还原，组织，整理和数据增强，组成batch输出
def create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                       batch_queue_capacity, num_batch_queue_threads,
                       prefetch_queue_capacity, data_augmentation_options):
    """Sets up reader, prefetcher and returns input queue.

    Args:
      batch_size_per_clone: batch size to use per clone.
      create_tensor_dict_fn: function to create tensor dictionary.
      batch_queue_capacity: maximum number of elements to store within a queue.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                               assembled batches.
      data_augmentation_options: a list of tuples, where each tuple contains a
        data augmentation function and a dictionary containing arguments and their
        values (see preprocessor.py).

    Returns:
      input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
        (which hold images, boxes and targets).  To get a batch of tensor_dicts,
        call input_queue.Dequeue().
    """
    tensor_dict = create_tensor_dict_fn()

    tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
        tensor_dict[fields.InputDataFields.image], 0)

    images = tensor_dict[fields.InputDataFields.image]
    float_images = tf.to_float(images)
    tensor_dict[fields.InputDataFields.image] = float_images

    include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks in tensor_dict)
    include_keypoints = (fields.InputDataFields.groundtruth_keypoints in tensor_dict)
    if data_augmentation_options:
        tensor_dict = preprocessor.preprocess(
            tensor_dict,
            data_augmentation_options,
            func_arg_map=preprocessor.get_default_func_arg_map(
                include_instance_masks=include_instance_masks,
                include_keypoints=include_keypoints))

    input_queue = batcher.BatchQueue(
        tensor_dict,
        batch_size=batch_size_per_clone,
        batch_queue_capacity=batch_queue_capacity,
        num_batch_queue_threads=num_batch_queue_threads,
        prefetch_queue_capacity=prefetch_queue_capacity)
    return input_queue


def get_inputs(input_queue, num_classes, merge_multiple_label_boxes=False):
    """Dequeues batch and constructs inputs to object detection model.

    Args:
      input_queue: BatchQueue object holding enqueued tensor_dicts.
      num_classes: Number of classes.
      merge_multiple_label_boxes: Whether to merge boxes with multiple labels
        or not. Defaults to false. Merged boxes are represented with a single
        box and a k-hot encoding of the multiple labels associated with the
        boxes.

    Returns:
      images: a list of 3-D float tensor of images.
      image_keys: a list of string keys for the images.
      locations_list: a list of tensors of shape [num_boxes, 4]
        containing the corners of the groundtruth boxes.
      classes_list: a list of padded one-hot tensors containing target classes.
      masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
        image_width] containing instance masks for objects if present in the
        input_queue. Else returns None.
      keypoints_list: a list of 3-D float tensors of shape [num_boxes,
        num_keypoints, 2] containing keypoints for objects if present in the
        input queue. Else returns None.
    """
    # BatchQueue.dequeue()返回一个list
    # list的元素是dict，长度是batch_size
    read_data_list = input_queue.dequeue()
    # 正类的编号从1开始，需要先改为从0开始
    label_id_offset = 1

    def extract_images_and_targets(read_data):
        """Extract images and targets from the input dict."""
        image = read_data[fields.InputDataFields.image]
        key = ''
        if fields.InputDataFields.source_id in read_data:
            key = read_data[fields.InputDataFields.source_id]
        location_gt = read_data[fields.InputDataFields.groundtruth_boxes]

        classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes], tf.int32)
        classes_gt -= label_id_offset
        if merge_multiple_label_boxes:
            location_gt, classes_gt, _ = util_ops.merge_boxes_with_multiple_labels(
                location_gt, classes_gt, num_classes)
        else:
            classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt, depth=num_classes, left_pad=0)
        masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
        keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
        if (merge_multiple_label_boxes and (
                masks_gt is not None or keypoints_gt is not None)):
            raise NotImplementedError('Multi-label support is only for boxes.')
        return image, key, location_gt, classes_gt, masks_gt, keypoints_gt
    # Python
    # 1.map/reduce功能
    # map(function, iterator) return iterator:
    # 将iterator的值依次传入函数处理，结果组成新的iterator
    # reduce(function, iterator) return value:
    # 将iterator的值依次传入函数处理，结果返回iterator中，得到最后依次迭代的结果
    # 2.zip功能
    # 当zip()有一个iterable时，将每个元素组成单独的tuple，所有的tuple组成一个zip类型
    # 当zip()有多个iterable时，将每个iterable的元素依次组成tuple，所有的tuple组成一个zip类型，iterable形状不同时，按最小值处理
    # *zip是zip的逆过程，将zip的结果还原
    # 3.星号表达式
    # *作用在可迭代对象时，或者变量
    return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn, train_config):
    """Creates loss function for a DetectionModel.

    Args:
      input_queue: BatchQueue object holding enqueued tensor_dicts.
      create_model_fn: A function to create the DetectionModel.
      train_config: a train_pb2.TrainConfig protobuf.
    """
    # 初始化模型框架
    detection_model = create_model_fn()
    # 得到标准化数据
    (images, _, groundtruth_boxes_list, groundtruth_classes_list,
     groundtruth_masks_list, groundtruth_keypoints_list) = get_inputs(
        input_queue,
        detection_model.num_classes,
        train_config.merge_multiple_label_boxes)

    # 数据预处理，resize和normalize
    images = [detection_model.preprocess(image) for image in images]
    images = tf.concat(images, 0)

    if any(mask is None for mask in groundtruth_masks_list):
        groundtruth_masks_list = None
    if any(keypoints is None for keypoints in groundtruth_keypoints_list):
        groundtruth_keypoints_list = None
    # 模型内部导入标注信息
    detection_model.provide_groundtruth(groundtruth_boxes_list,
                                        groundtruth_classes_list,
                                        groundtruth_masks_list,
                                        groundtruth_keypoints_list)
    # 得到预测结果
    prediction_dict = detection_model.predict(images)
    # 得到预测损失
    losses_dict = detection_model.loss(prediction_dict)
    # 加入losses collection
    for loss_tensor in losses_dict.values():
        tf.losses.add_loss(loss_tensor)


def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir):
    """Training function for detection models.

    Args:
      create_tensor_dict_fn: a function to create a tensor input dictionary.
      create_model_fn: a function that creates a DetectionModel and generates losses.
      train_config: a train_pb2.TrainConfig protobuf.
      master: BNS name of the TensorFlow master to use.
      task: The task id of this training instance.
      num_clones: The number of clones to run per machine.
      worker_replicas: The number of work replicas to train with.
      clone_on_cpu: True if clones should be forced to run on CPU.
      ps_tasks: Number of parameter server tasks.
      worker_job_name: Name of the worker job.
      is_chief: Whether this replica is the chief replica.
      train_dir: Directory to write checkpoints and training summaries to.
    """
    # 实例化了一个目标检测模型
    detection_model = create_model_fn()
    # 是一个list,包含各种指定的数据增强函数
    data_augmentation_options = [preprocessor_builder.build(step) for step in train_config.data_augmentation_options]

    with tf.Graph().as_default():
        # Build a configuration specifying multi-GPU and multi-replicas.
        # 这是一个类对象 方便快速配置分布式参数
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=num_clones,
            clone_on_cpu=clone_on_cpu,
            replica_id=task,
            num_replicas=worker_replicas,
            num_ps_tasks=ps_tasks,
            worker_job_name=worker_job_name)

        # Place the global step on the device storing the variables.
        with tf.device(deploy_config.variables_device()):
            # global_step = slim.create_global_step()
            global_step = tf.train.create_global_step()

        with tf.device(deploy_config.inputs_device()):
            input_queue = create_input_queue(
                train_config.batch_size // num_clones, create_tensor_dict_fn,
                train_config.batch_queue_capacity,
                train_config.num_batch_queue_threads,
                train_config.prefetch_queue_capacity, data_augmentation_options)

        # Gather initial summaries.
        # TODO(rathodv): See if summaries can be added/extracted from global tf
        # collections so that they don't have to be passed around.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_summaries = set([])
        # 创建包含loss步骤的模型用于train
        model_fn = functools.partial(_create_losses,
                                     create_model_fn=create_model_fn,
                                     train_config=train_config)
        # 什么是clones, 是指同步或异步以Between-Graph的模式进行训练，为每个GPU做出相同的部署
        clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
        first_clone_scope = clones[0].scope
        # 只给出第一块GPU上的更新操作节点
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by model_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        # 配置优化器
        with tf.device(deploy_config.optimizer_device()):
            training_optimizer = optimizer_builder.build(train_config.optimizer, global_summaries)
        # 配置同步模式 优化器
        sync_optimizer = None
        if train_config.sync_replicas:
            # 同步更新
            training_optimizer = tf.train.SyncReplicasOptimizer(
                training_optimizer,
                replicas_to_aggregate=train_config.replicas_to_aggregate,
                total_num_replicas=train_config.worker_replicas)
            sync_optimizer = training_optimizer
        # 准备参数映射map 初始化函数
        # Create ops required to initialize the model from a given checkpoint.
        init_fn = None
        if train_config.fine_tune_checkpoint:
            # 想要获得的变量
            var_map = detection_model.restore_map(
                from_detection_checkpoint=train_config.from_detection_checkpoint)
            # 可以得到的参数--ckpt文件中对应上的参数
            available_var_map = (variables_helper.get_variables_available_in_checkpoint(
                var_map, train_config.fine_tune_checkpoint))

            init_saver = tf.train.Saver(available_var_map)

            def initializer_fn(sess):
                init_saver.restore(sess, train_config.fine_tune_checkpoint)

            init_fn = initializer_fn

        # 管理total loss和update op
        with tf.device(deploy_config.optimizer_device()):
            # total_loss是单个运算设备的loss_dict的总和,如果是多个GPU训练，那就取平均值
            # grads_and_vars是优化器计算得到的梯度 A list of (gradient, variable) pairs
            total_loss, grads_and_vars = \
                model_deploy.optimize_clones(clones, training_optimizer, regularization_losses=None)
            # Checks a tensor for NaN and Inf values.
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')
            # 为偏置参数的梯度 乘以一个系数
            # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
            if train_config.bias_grad_multiplier:
                biases_regex_list = ['.*/biases']
                grads_and_vars = variables_helper.multiply_gradients_matching_regex(
                    grads_and_vars,
                    biases_regex_list,
                    multiplier=train_config.bias_grad_multiplier)
            # 冻结指定参数的梯度 改为0
            # Optionally freeze some layers by setting their gradients to be zero.
            if train_config.freeze_variables:
                grads_and_vars = \
                    variables_helper.freeze_gradients_matching_regex(grads_and_vars, train_config.freeze_variables)
            # 梯度裁剪，防止梯度爆炸，手段就是控制梯度的最大范式，内部封装了tf.clip_by_norm
            # t = t * clip_norm / l2norm(t)
            # Optionally clip gradients
            if train_config.gradient_clipping_by_norm > 0:
                with tf.name_scope('clip_grads'):
                    grads_and_vars = slim.learning.clip_gradient_norms(
                        grads_and_vars, train_config.gradient_clipping_by_norm)
            # tf.train.optimizer的apply_gradients方法，开始更新参数
            # Create gradient updates.
            grad_updates = training_optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            update_ops.append(grad_updates)
            # tf.group() 星号表达式
            update_op = tf.group(*update_ops)
            # 指定依赖关系tf.control_dependencies,先执行update_op节点的操作，才能执行train_tensor节点的操作
            with tf.control_dependencies([update_op]):
                # tf.identity() 创建一个新节点train_tensor
                # train_tensor节点的tensor和total_loss节点的tensor是同一个，但是值不同，
                # train_tensor节点的tensor值，取update_op之后的值
                # 优点：获取每次更新之后的loss值，total_loss获取初始loss值，update_op获取的值太多
                train_tensor = tf.identity(total_loss, name='train_op')

        # Add summaries.
        for model_var in slim.get_model_variables():
            global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
        for loss_tensor in tf.losses.get_losses():
            global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
        global_summaries.add(tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        # '|'按位或 运算 再赋值， 类似于 +=
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        summaries |= global_summaries
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # session_config.gpu_options.allow_growth = True 自由分配GPU占用比例

        # Save checkpoints regularly.
        # 按固定时间间隔保存模型 default=1000 没有用，不用管，
        keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, max_to_keep=None)

        # 以上内容在配置分布式训练参数
        # slim.learining.train内封装了tensorflow的supervisor通过管理session来进行训练
        # supervisor内部有一个线程 专门用来启动saver,每10分钟启动一次
        # tf.contrib.slim.python.slim.learning.tarin()
        slim.learning.train(
            train_tensor,
            logdir=train_dir,
            master=master,
            is_chief=is_chief,
            session_config=session_config,
            startup_delay_steps=train_config.startup_delay_steps,
            init_fn=init_fn,
            summary_op=summary_op,
            number_of_steps=(train_config.num_steps if train_config.num_steps else None),
            save_summaries_secs=120,
            sync_optimizer=sync_optimizer,
            saver=saver)
