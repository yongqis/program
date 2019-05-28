"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

from object_detection.simple import model_hparams
from object_detection.simple import model_lib

flags.DEFINE_string('model_dir', None, 'Path to output model, event and checkpoint files directory.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')

flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of every n(1) eval input examples.')
flags.DEFINE_string('hparams_overrides', None, ' a string containing comma-separated hparam_name=value pairs.')
flags.DEFINE_string('checkpoint_dir', None,  'If is provided, eval-only mode, writing resulting metrics to model_dir.')
flags.DEFINE_boolean('run_once', False, 'If in eval-only mode, whether to run one round of eval or continuously.')
FLAGS = flags.FLAGS


def main(unused_argv):
    """
    已简化内容：
    1、删除了tpu参数相关语句
    2、删除eval on train data相关语句
    3、eval_input_fn 只有一个，不再是多个组成的list
    4、删除train_steps参数，可以从config内获取相应数据
    待解决问题：
    1、理清参数加载的相关内容：
        1.model_fn中tf.train.init_from_checkpoint()和 tf.estimator.RunConfig() 两者的加载功能怎样解决冲突
        2.predict和eval时参数如何加载
        3.继续训练时是否会加载global_step
    2、input_fn的高级多线程操作：
        1.tf.data的文件list多线程
        2.tf.dataset的apply()函数
    """
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')

    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
    # 创建estimator和输入函数 得到一个字典
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples)
    # 单独提取数据
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fn = train_and_eval_dict['eval_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    # 如果给出ckpt路径，那么对目标ckpt模型进行eval，否则train and eval
    if FLAGS.checkpoint_dir:
        # 验证1轮或多轮
        if FLAGS.run_once:
            estimator.evaluate(eval_input_fn, checkpoint_path=tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, eval_input_fn, train_steps, 'validation_data')
    else:
        train_spec, eval_spec = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fn,
            predict_input_fn,
            train_steps)
        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()
