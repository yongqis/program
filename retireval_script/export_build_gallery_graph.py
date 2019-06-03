#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import logging
import tempfile
import tensorflow as tf

from object_detection_updata.protos import pipeline_pb2
from object_detection_updata.builders import model_builder

from google.protobuf import text_format

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.training import saver as saver_lib
from tensorflow.core.protobuf import rewriter_config_pb2

slim = tf.contrib.slim

flags = tf.flags

flags.DEFINE_string('pipeline_config_path', '/home/hnu/workspace/syq/retrieval/data/all_class/saved_model/5-7/pipeline.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', '/home/hnu/workspace/syq/retrieval/data/all_class/saved_model/5-7/model.ckpt-58112',
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', '/home/hnu/workspace/syq/retrieval/data/all_class/saved_model', 'Path to write outputs.')

# flags.mark_flag_as_required('pipeline_config_path')
# flags.mark_flag_as_required('trained_checkpoint_prefix')
# flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def _image_tensor_input_placeholder(input_shape=None):
    """Returns input placeholder and a 4-D uint8 image tensor."""
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(dtype=tf.uint8, shape=input_shape, name='image_tensor')
    return input_tensor


# TODO: Replace with freeze_graph.freeze_graph_with_def_protos when
# newer version of Tensorflow becomes more common.
def freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        clear_devices,
        initializer_nodes,
        optimize_graph=True,
        variable_names_blacklist=''):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError('Input checkpoint "' + input_checkpoint + '" does not exist!')

    if not output_node_names:
        raise ValueError('You must supply the name of a node to --output_node_names.')

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')

        if optimize_graph:
            logging.info('Graph Rewriter optimizations enabled')
            rewrite_options = rewriter_config_pb2.RewriterConfig(layout_optimizer=True)
            rewrite_options.optimizers.append('pruning')
            rewrite_options.optimizers.append('constfold')
            rewrite_options.optimizers.append('layout')

            graph_options = tf.GraphOptions(rewrite_options=rewrite_options, infer_shapes=True)

        else:
            logging.info('Graph Rewriter optimizations disabled')
            graph_options = tf.GraphOptions()

        config = tf.ConfigProto(graph_options=graph_options)
        with session.Session(config=config) as sess:
            if input_saver_def:
                saver = saver_lib.Saver(saver_def=input_saver_def)
                saver.restore(sess, input_checkpoint)

            else:
                var_list = {}
                reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    try:
                        tensor = sess.graph.get_tensor_by_name(key + ':0')
                    except KeyError:
                        # This tensor doesn't exist in the graph (for example it's
                        # 'global_step' or a similar housekeeping element) so skip it.
                        continue
                    var_list[key] = tensor
                saver = saver_lib.Saver(var_list=var_list)
                saver.restore(sess, input_checkpoint)
                if initializer_nodes:
                    sess.run(initializer_nodes)

            variable_names_blacklist = (variable_names_blacklist.split(',')
                                        if variable_names_blacklist else None)
            # 将计算图中的变量及其取值转换为常量，并且只保留需要的节点
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_blacklist=variable_names_blacklist)

    return output_graph_def


def replace_variable_values_with_moving_averages(graph,
                                                 current_checkpoint_file,
                                                 new_checkpoint_file):
    """Replaces variable values in the checkpoint with their moving averages.

    If the current checkpoint has shadow variables maintaining moving averages of
    the variables defined in the graph, this function generates a new checkpoint
    where the variables contain the values of their moving averages.

    Args:
      graph: a tf.Graph object.
      current_checkpoint_file: a checkpoint containing both original variables and
        their moving averages.
      new_checkpoint_file: file path to write a new checkpoint.
    """
    with graph.as_default():
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        ema_variables_to_restore = variable_averages.variables_to_restore()
        with tf.Session() as sess:
            read_saver = tf.train.Saver(ema_variables_to_restore)
            read_saver.restore(sess, current_checkpoint_file)
            write_saver = tf.train.Saver()
            write_saver.save(sess, new_checkpoint_file)


def _add_output_tensor_nodes(postprocessed_tensors, output_collection_name='inference_op'):
    """Adds output nodes for detection boxes and scores.
    Args:
      postprocessed_tensors:
      output_collection_name: Name of collection to add output tensors to.

    Returns:
      A tensor dict containing the added output tensor nodes.
    """
    outputs = {}
    outputs['embeddings'] = tf.identity(postprocessed_tensors, name='embeddings')

    # 将tensor加入集合，outputs默认返回keys
    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])

    return outputs


def _write_frozen_graph(frozen_graph_path, frozen_graph_def):
    """Writes frozen graph to disk.

    Args:
      frozen_graph_path: Path to write inference graph.
      frozen_graph_def: tf.GraphDef holding frozen graph.
    """
    with gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
    logging.info('%d ops in the final graph.', len(frozen_graph_def.node))


def main(_):
    output_directory = FLAGS.output_directory
    trained_checkpoint_prefix = FLAGS.trained_checkpoint_prefix
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        text_format.Merge(f.read(), pipeline_config)  # 默认值与配置值融合

    optimize_graph = True
    output_collection_name = 'inference_op'
    use_moving_averages = pipeline_config.eval_config.use_moving_averages

    detection_model = model_builder.build(pipeline_config.model, is_training=False, is_building=True)
    # 0.准备文件保存路径，初始值的检查
    # 确保文件路径
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory, 'frozen_build_gallery_graph.pb')
    # 1.计算图inference阶段
    # 得到input_tensor并float化
    placeholder_tensors = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 3], name='image_tensor')
    inputs = tf.to_float(placeholder_tensors)
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    # 2.将原来保存输出结果的字典，重新返回一个字典outputs，并将新字典的tensor加入到集合
    outputs = _add_output_tensor_nodes(output_tensors, output_collection_name)
    # Add global step to the graph.
    # return The global step tensor.
    tf.train.get_or_create_global_step()
    # 3.如果使用滑动平均，更新ckpt保存的值
    if use_moving_averages:
        temp_checkpoint_file = tempfile.NamedTemporaryFile()
        replace_variable_values_with_moving_averages(tf.get_default_graph(),
                                                     trained_checkpoint_prefix,
                                                     temp_checkpoint_file.name)
        checkpoint_to_use = temp_checkpoint_file.name
    else:
        checkpoint_to_use = trained_checkpoint_prefix  # 训练后保存的ckpt文件路径
    # 4.序列化saver.as_saver_def()--tf.train.Saver(saver_def)
    saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()
    output_node_names = ','.join(outputs.keys())
    # 5.序列化一个精简后的计算图
    frozen_graph_def = freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=checkpoint_to_use,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        optimize_graph=optimize_graph,
        initializer_nodes='')
    # 6.保存该精简的计算图
    _write_frozen_graph(frozen_graph_path, frozen_graph_def)


if __name__ == '__main__':
    tf.app.run()