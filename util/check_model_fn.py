import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from util.input_fn import train_input_fn
from util.update_util import clip_gradient_norms
from object_detection_updata.utils import config_util
from object_detection_updata.utils import ops as util_ops
from object_detection_updata.utils import variables_helper
from object_detection_updata.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from object_detection_updata.builders import model_builder


pipeline_config_path = '/home/hnu/workspace/syq/retrieval/data/faster_rcnn_inception_resnet_v2.config'
save_model_dir = ''
configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
model_config = configs['model']
train_config = configs['train_config']
input_config = configs['train_input_config']

# train_input_fn 得到训练数据文件解析结果作为模型的input data
batch_image, batch_gt_label = train_input_fn(input_config.tf_record_input_reader.input_path, input_config)

# build 首先初始化模型的辅助工具 以及 初始化一个模型
detection_model = model_builder.build(model_config=model_config, is_training=True)  # 初始化模型的辅助工具
# label 处理并送入模型保存
num_classes = detection_model.num_classes
batch_boxes, batch_classes = batch_gt_label
classes = tf.cast(batch_classes[0], tf.int32)
classes -= 1
classes_gt = util_ops.padded_one_hot_encoding(indices=classes, depth=num_classes, left_pad=0)

boxes_gt = batch_boxes[0]
# batch中每个图片的label数量不同，因此label不能组成batch，用list
groundtruth_boxes_list, groundtruth_classes_list = [boxes_gt], [classes_gt]

detection_model.provide_groundtruth(groundtruth_boxes_list, groundtruth_classes_list)

# 数据类型处理
# batch维度的大小固定下来, resize和normalize Tensor.get_shape.as_list只能读取静态shape
image = batch_image[0]
batch_images = tf.expand_dims(image, 0)
batch_images = tf.cast(batch_images, tf.float32)
# 构建计算图
batch_images = detection_model.preprocess(batch_images)  # 预处理
rpn_prediction_dict = detection_model.predict_rpn(batch_images)  # 第一阶段预测结果
rpn_detection_dict = detection_model.postprocess_rpn(rpn_prediction_dict)  # 第一阶段预测结果后处理
rpn_loss_dict = detection_model.loss_rpn(rpn_prediction_dict)
for loss_tensor in rpn_loss_dict.values():
    tf.losses.add_loss(loss_tensor)

if not model_config.first_stage_only:
    final_prediction_dict = detection_model.predict_second_stage(rpn_detection_dict)  # 第二阶段预测结果
    final_detection_dict = detection_model.postprocess_box_classifier(final_prediction_dict)
    final_loss_dict = detection_model.loss_box_classifier(final_prediction_dict)
    for loss_tensor in final_loss_dict.values():
        tf.losses.add_loss(loss_tensor)

total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.Losses))

# 挑选出中间节点 可以查看数据
# proposal_boxes = rpn_detection_dict['proposal_boxes']
# proposal_boxes = final_prediction_dict['proposal_boxes']
# triplet_label = final_loss_dict['triplet_label']

# 定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
# 获得需要更新的变量及其梯度
grads_and_vars = optimizer.compute_gradients(total_loss)
# 梯度更新前 3个的约束条件
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
# 定义参数更新操作
global_step = tf.train.get_or_create_global_step()
grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
# 更新操作汇总--BN/moving mean需要单独的更新操作
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops.append(grad_updates)
update_op = tf.group(*update_ops)  # tf.group() 星号表达式
# 指定依赖关系--先执行update_op节点的操作，才能执行train_tensor节点的操作
with tf.control_dependencies([update_op]):
    train_tensor = tf.identity(total_loss, name='train_op')  # tf.identity()

# 准备加载参数的映射字典
# 首先加载训练过程中保存的最新模型，如果没有再加载config中指定的文件
ckpt = tf.train.get_checkpoint_state(save_model_dir)
if ckpt and ckpt.model_checkpoint_path:
    model_path = ckpt.model_checkpoint_path
else:
    model_path = train_config.fine_tune_checkpoint
var_map = detection_model.restore_map(from_detection_checkpoint=train_config.from_detection_checkpoint)
available_var_map = variables_helper.get_variables_available_in_checkpoint(var_map, model_path)
init_saver = tf.train.Saver(available_var_map)

# 创建会话 运行计算图
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    init_saver.restore(sess, train_config.fine_tune_checkpoint)
    save_model_path = os.path.join(save_model_dir, 'model.ckpt')

    while True:
        try:
            loss, step = sess.run([image, train_tensor, global_step])
            if step % 100 == 0:
                print("After %d training steps, loss on training batch is %g" % (step, loss))
            if step % 1000 == 0:
                init_saver.save(sess, save_path=save_model_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('finish')
            break

