import os
import json
import functools
import tensorflow as tf

from object_detection_updata.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
flags.DEFINE_string('train_dir', r'D:\Pycharm\joint_retrieval\data\all_class\saved_model',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('pipeline_config_path',
                    r'D:\Pycharm\joint_retrieval\data\all_class\faster_rcnn_inception_v2_pets.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config file. If provided, other configs are ignored')
FLAGS = flags.FLAGS


def main(_):
    assert FLAGS.train_dir, '`train_dir` is missing.'
    tf.gfile.MakeDirs(FLAGS.train_dir)
    configs = config_util.get_configs_from_pipeline_file(FLAGS.pipeline_config_path)
    tf.gfile.Copy(FLAGS.pipeline_config_path, os.path.join(FLAGS.train_dir, 'pipeline.config'), overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']






if __name__ == '__main__':
    tf.app.run()
