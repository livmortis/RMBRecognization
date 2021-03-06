import datetime
import os
import sys
import time

import tensorflow as tf

sys.path.append(os.getcwd())
from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider


'''
--learning_rate  设置学习率
--max_steps   设置epoch
--restore   是否读取模型

'''
# mianzhi = '_0_1'
# mianzhi = '_0_2'
# mianzhi = '_0_5'
# mianzhi = '_1'
# mianzhi = '_2'
# mianzhi = '_5'
# mianzhi = '_10'
# mianzhi = '_50'
# mianzhi = '_100'
mianzhi = '_tight1000'  # 2019.6.27, 重新手标1000个紧凑标签。

lrboundaries =    [55000.0, 57000.0, 59000.0, 61000.0, 63000.0, 65000.0, 67000.0, 69000.0, 71000.0, 73000.0, 75000.0, 77000.0, 79000.0, 82000.0, 85000.0, 88000.0, 91000.0, 95000.0,  100000.0 ,  110000.0]
lrvalues =      [1e-6,    8e-7,    6e-7,   4e-7,     2e-7,   9e-8,     7e-8,   5e-8,    4e-8,   3e-8,     2e-8,    1e-8,     9e-9,   8e-9,   7e-9,      6e-9,     5e-9,  4e-9,   3e-9,      2e-9,       1e-9]

# lrboundaries =    [55000.0, 57000.0, 59000.0, 61000.0, 63000.0, 65000.0, 67000.0, 69000.0, 71000.0, 73000.0, 75000.0, 77000.0, 79000.0, 82000.0, 85000.0, 88000.0 ]
# lrvalues =      [1e-6,    8e-7,    6e-7,   4e-7,     2e-7,   9e-8,     7e-8,   5e-8,    3e-8,   1e-8,     9e-9,    8e-9,     7e-9,   6e-9,   4e-9,      2e-9,     1e-9]

tf.app.flags.DEFINE_float('learning_rate', 1e-6, '')
# tf.app.flags.DEFINE_integer('max_steps', 50000, '')
# tf.app.flags.DEFINE_integer('max_steps', 91000, '')    #xzy
tf.app.flags.DEFINE_integer('max_steps', 130000, '')    #xzy
tf.app.flags.DEFINE_boolean('restore', True, '')


tf.app.flags.DEFINE_integer('decay_steps', 30000, '')   #xzy 弃用
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_readers', 4, '')
tf.app.flags.DEFINE_string('gpu', '0', '')

# tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../../../dataset_formal/detect_data/CTPNData/checkpoints_mlt/', '')

# tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_string('logs_path', '../../../dataset_formal/detect_data/CTPNData/logs_mlt/', '')

# tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')
tf.app.flags.DEFINE_string('pretrained_model_path', '../../../dataset_formal/detect_data/CTPNData/vgg_16.ckpt', '')

# tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 50, '')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime + mianzhi)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox')
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    # learning_rate = FLAGS.learning_rate     #xzy 为了避免加载预训练时，强制加载预训练的学习率（lr= 1e-5,太小了），而手动设置lr。
    learning_rate = tf.train.piecewise_constant(global_step,lrboundaries,lrvalues)    #xzy 1800训练时加入学习率策略
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    gpu_id = int(FLAGS.gpu)
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox,
                                                                                 input_im_info)
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime + mianzhi, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            # restore_step = int(ckpt.split('/')[-1].split('.')[0].split('_')[-1])
            restore_step = int(ckpt.split('/')[-1].split('.')[0].split('_')[1])     #xzy 1800标签版本，ckpt文件命名更改。
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers)
        start = time.time()
        for step in range(restore_step, FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                              feed_dict={input_image: data[0],
                                                         input_bbox: data[1],
                                                         input_im_info: data[2]})

            summary_writer.add_summary(summary_str, global_step=step)

            # if step != 0 and step % FLAGS.decay_steps == 0:
            #     sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))       #xzy 手动修改学习率（源码为3万epoch改一次..）

            # if step % 10 == 0:
            if step % 1 == 0:       #xzy 每个epoch打印一次
                # avg_time_per_step = (time.time() - start) / 10
                avg_time_per_step = (time.time() - start)   #xzy 每个epoch打印一次
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    step, ml, tl, avg_time_per_step, learning_rate.eval()))     #xzy 6.24更新，添加lr策略后，learning_rate又成了tensor
                # print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                #     step, ml, tl, avg_time_per_step, learning_rate))    #xzy 手动设置学习率

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) +mianzhi+'.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))


if __name__ == '__main__':
    tf.app.run()
