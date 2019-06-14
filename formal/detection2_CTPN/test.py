# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('test_data_path', '../../../dataset_warm_up/train_data/', '')  #检测比赛训练集
# tf.app.flags.DEFINE_string('test_data_path', '../../../dataset_warm_up/public_test_data/', '')  #检测比赛测试集

tf.app.flags.DEFINE_string('output_path', '../../../dataset_formal/detect_data/polyImg_CTPN_train', '') #训练集poly输出路径
# tf.app.flags.DEFINE_string('output_path', '../../../dataset_formal/detect_data/polyImg_CTPN_test', '')  #测试集poly输出路径

tf.app.flags.DEFINE_string('gpu', '0', '')

# tf.app.flags.DEFINE_string('checkpoint_path', '../checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../../../dataset_formal/detect_data/CTPNData/checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


npy_path = '../../../dataset_formal/detect_data/CTPNData/'

# tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
# tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
# tf.app.flags.DEFINE_string('gpu', '0', '')
# tf.app.flags.DEFINE_string('checkpoint_path', '../../../dataset_formal/detect_data/CTPNData/checkpoints_mlt/', '')
# FLAGS = tf.app.flags.FLAGS



def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        filenames.sort()                                #xzy 加入排序，多gpu同时预测
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            ii = 0
            for im_fn in im_fn_list[:3]:          #修改这里
                if ii ==0:
                    im_fn = "../../../dataset_warm_up/train_data/13DGOXW9.jpg"      #测试
                elif ii==1:
                    im_fn = "../../../dataset_warm_up/train_data/WBNGQ9R7.jpg"  # 测试
                else:
                    im_fn = "../../../dataset_warm_up/train_data/013MNV9B.jpg"  # 测试

                ii += 1
                print(str(ii)+'==============='+str(ii))
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})
                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]   # 每张图片N个poly，textsegs是这些poly的四个坐标。

                textdetector = TextDetector(DETECT_MODE='H')

                try:

                    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])   #xzy 方法内部已修改，只显示一个框
                    boxes = np.array(boxes, dtype=np.int)

                    cost_time = (time.time() - start)
                    print("cost time: {:.2f}s".format(cost_time))

                    for i, box in enumerate(boxes):
                        # cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                        #               thickness=2)
                        img = img[int(box[1]): int(box[5]), int(box[0]): int(box[2]) ]    # xzy 裁剪

                    img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])
                except Exception as e:
                    immmm = cv2.imread("../../../dataset_warm_up/train_data/0BRO7XVG.jpg")      #xzy 可能WBNGQ9R7.jpg出错
                    cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), immmm[:, :, ::-1])
                    print(str(im_fn)+" is broken!!!!!!!!")
                    #TODO  写一个txt记录错误图片名



                # with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",   #xzy 取消写txt
                #           "w") as f:
                #     for i, box in enumerate(boxes):
                #         line = ",".join(str(box[k]) for k in range(8))
                #         line += "," + str(scores[i]) + "\r\n"
                #         f.writelines(line)


if __name__ == '__main__':
    tf.app.run()
