"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

from simple_train import model_hparams
from simple_train import model_lib

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

    已解决问题：
    首先弄清tf.estimator.Estimator()对象在train/eval/predict时，是运行的
    其内部使用tf.train.MonitorSession()了来创建一个ChiefSessionCreator，list of hook
    创建会话后，首先加载model_dir内的ckpt模型，如果成功就不再执行初始化操作init_op()
    最后开始sess.run()管理计算图
    1、参数加载的相关内容：
        1.model_fn中tf.train.init_from_checkpoint()和 tf.estimator.RunConfig() 两者的加载参数时怎样解决冲突
        答：在构建计算图时，定义了tf.train.init_from_checkpoint()，在会话run init_op时，覆盖（override）全局初始化操作加载参数
        2.predict和eval时参数如何加载
        答：从model_dir
        3.继续训练时是否会加载global_step
        答：从model_dir加载参数时，会加载global_step
    待解决问题：
    1、input_fn的高级多线程操作：
        1.tf.data的文件list多线程
        2.tf.dataset的apply()函数
    2、eval时exporter对象的实现
    3、深入理解tf.train.MonitorSession()/hook/scaffold
    """
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')

    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
    # 创建estimator和输入函数 得到一个字典
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples)

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
