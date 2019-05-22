import logging
import os
import tempfile
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder

slim = tf.contrib.slim


def _image_tensor_input_placeholder(input_shape=None):
    """Returns input placeholder and a 4-D uint8 image tensor."""
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(dtype=tf.uint8, shape=input_shape, name='image_tensor')
    return input_tensor, input_tensor


def _tf_example_input_placeholder():
    """Returns input that accepts a batch of strings with tf examples.

    Returns:
      a tuple of input placeholder and the output decoded images.
    """
    batch_tf_example_placeholder = tf.placeholder(
        tf.string, shape=[None], name='tf_example')

    def decode(tf_example_string_tensor):
        tensor_dict = tf_example_decoder.TfExampleDecoder().decode(
            tf_example_string_tensor)
        image_tensor = tensor_dict[fields.InputDataFields.image]
        return image_tensor

    return (batch_tf_example_placeholder,
            tf.map_fn(decode,
                      elems=batch_tf_example_placeholder,
                      dtype=tf.uint8,
                      parallel_iterations=32,
                      back_prop=False))


def _encoded_image_string_tensor_input_placeholder():
    """Returns input that accepts a batch of PNG or JPEG strings.

    Returns:
      a tuple of input placeholder and the output decoded images.
    """
    batch_image_str_placeholder = tf.placeholder(
        dtype=tf.string,
        shape=[None],
        name='encoded_image_string_tensor')

    def decode(encoded_image_string_tensor):
        image_tensor = tf.image.decode_image(encoded_image_string_tensor, channels=3)
        image_tensor.set_shape((None, None, 3))
        return image_tensor

    return (batch_image_str_placeholder,
            tf.map_fn(
                decode,
                elems=batch_image_str_placeholder,
                dtype=tf.uint8,
                parallel_iterations=32,
                back_prop=False))


input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor': _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder,
}


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

    Adds the following nodes for output tensors -
      * num_detections: float32 tensor of shape [batch_size].
      * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
        containing detected boxes.
      * detection_scores: float32 tensor of shape [batch_size, num_boxes]
        containing scores for the detected boxes.
      * detection_classes: float32 tensor of shape [batch_size, num_boxes]
        containing class predictions for the detected boxes.
      * detection_masks: (Optional) float32 tensor of shape
        [batch_size, num_boxes, mask_height, mask_width] containing masks for each
        detection box.

    Args:
      postprocessed_tensors: a dictionary containing the following fields
        'detection_boxes': [batch, max_detections, 4]
        'detection_scores': [batch, max_detections]
        'detection_classes': [batch, max_detections]
        'detection_masks': [batch, max_detections, mask_height, mask_width] (optional).
        'num_detections': [batch]
      output_collection_name: Name of collection to add output tensors to.

    Returns:
      A tensor dict containing the added output tensor nodes.
    """

    # 将预测结果的类别id加1
    boxes = postprocessed_tensors.get('detection_boxes')
    scores = postprocessed_tensors.get('detection_scores')
    num_proposals = postprocessed_tensors.get('num_detections')
    obj_embeddings = postprocessed_tensors.get('box_embeddings')
    outputs = {}
    outputs['detection_boxes'] = tf.identity(boxes, name='detection_boxes')
    outputs['detection_scores'] = tf.identity(scores, name='detection_scores')
    outputs['num_detections'] = tf.identity(num_proposals, name='num_detections')
    outputs['box_embeddings'] = tf.identity(obj_embeddings, name='box_embeddings')
    # masks单独判断并处理
    masks = postprocessed_tensors.get('detection_masks')
    if masks is not None:
        outputs['detection_masks'] = tf.identity(masks, name='detection_masks')
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


def _write_saved_model(saved_model_path,
                       frozen_graph_def,
                       inputs,
                       outputs):
    """Writes SavedModel to disk.

    If checkpoint_path is not None bakes the weights into the graph thereby
    eliminating the need of checkpoint files during inference. If the model
    was trained with moving averages, setting use_moving_averages to true
    restores the moving averages, otherwise the original set of variables
    is restored.

    Args:
      saved_model_path: Path to write SavedModel.
      frozen_graph_def: tf.GraphDef holding frozen graph.
      inputs: The input image tensor to use for detection.
      outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    with tf.Graph().as_default():
        with session.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')

            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

            tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )
            builder.save()


def _write_graph_and_checkpoint(inference_graph_def,
                                model_path,
                                input_saver_def,
                                trained_checkpoint_prefix):
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        # tf.import_graph_def()--将序列化的计算图inference_graph_def,重新加入到当前默认计算图中
        tf.import_graph_def(inference_graph_def, name='')
        with session.Session() as sess:
            # Saver(saver_def=)
            saver = saver_lib.Saver(saver_def=input_saver_def, save_relative_paths=True)
            # 加载保存的模型
            saver.restore(sess, trained_checkpoint_prefix)
            # 重新保存
            saver.save(sess, model_path)


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            optimize_graph=True,
                            output_collection_name='inference_op'):
    """

    :param input_type: 输入的类型必须在input_placeholder_fn_map,默认image_tensor
    此dict里面有对应不同输入类型的处理函数 'image_tensor','encoded_image_string_tensor', 'tf_example',
    :param detection_model: 初始化过的一个模型，必须使用和训练模型时相同的config文件
    :param use_moving_averages: 是否使用滑动平均
    :param trained_checkpoint_prefix: 训练保存的模型参数，path/to/model.ckpt
    :param output_directory: 导出结果的目录路径
    :param additional_output_tensor_names:
    :param input_shape:
    :param optimize_graph:
    :param output_collection_name:
    :return:
    """
    # 0.准备文件保存路径，初始值的检查
    # 确保文件路径
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory, 'frozen_inference_graph.pb')

    # 1.计算图inference阶段
    placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type]()
    inputs = tf.to_float(input_tensors)
    # 按照模型的流程得到预测结果
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)
    # 2.将原来保存输出结果的字典，重新返回一个字典outputs，并将新字典的tensor加入到集合
    outputs = _add_output_tensor_nodes(postprocessed_tensors, output_collection_name)
    # Add global step to the graph.
    # return The global step tensor.
    tf.train.get_or_create_global_step()
    # 3.如果使用滑动平均，更新ckpt保存的值
    if use_moving_averages:
        temp_checkpoint_file = tempfile.NamedTemporaryFile()
        replace_variable_values_with_moving_averages(
            tf.get_default_graph(), trained_checkpoint_prefix,
            temp_checkpoint_file.name)
        checkpoint_to_use = temp_checkpoint_file.name
    else:
        checkpoint_to_use = trained_checkpoint_prefix  # 训练后保存的ckpt文件路径
    # 4.序列化saver.as_saver_def()
    saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()

    # 字符串的join方法和split方法
    if additional_output_tensor_names is not None:
        output_node_names = ','.join(outputs.keys() + additional_output_tensor_names)
    else:
        output_node_names = ','.join(outputs.keys())
    # 6.序列化一个精简后的计算图
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
    # 7.保存该精简的计算图
    _write_frozen_graph(frozen_graph_path, frozen_graph_def)


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           optimize_graph=True,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None):
    """Exports inference graph for the model specified in the pipeline config.

    Args:
    input_type: Type of input for the graph. Can be one of [`image_tensor`,
      `tf_example`].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    trained_checkpoint_prefix: Path to the trained checkpoint file.
    output_directory: Path to write outputs.
    input_shape: Sets a fixed shape for an `image_tensor` input. If not
      specified, will default to [None, None, None, 3].
    optimize_graph: Whether to optimize graph using Grappler.
    output_collection_name: Name of collection to add output tensors to.
      If None, does not add output tensors to a collection.
    additional_output_tensor_names: list of additional output
    tensors to include in the frozen graph.
    """
    # 根据配置文件的model部分初始化一个目标检查模型。class
    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    # 根据配置文件eval部分导出模型
    _export_inference_graph(input_type, detection_model,
                            pipeline_config.eval_config.use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory, additional_output_tensor_names,
                            input_shape, optimize_graph, output_collection_name)