# coding:utf-8
import glob
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

import tensorflow as tf

import pandas as pd
import os
from tqdm import tqdm
import PIL.Image as Image
import fdConfig
import torch.utils.data as Data
import torch
import cv2
import numpy as np
import fdConfig
from torchvision import transforms



tf.app.flags.DEFINE_string('training_data_path', '/Users/xzy/Documents/coder/ML/project/Self_east_ocr/train_image_Incidental_Scene_Text',
                           'training dataset to use')

tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')
tf.app.flags.DEFINE_string('geometry', 'RBOX',
                           'which geometry to generate, RBOX or QUAD')


FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)

        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags
    return im, polys, tags


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:      #xzy p1[0] 左上点x；  p1[1]左上点y
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)     #xzy p1：左上点x和 右上点x；   p2：左上点y和右上点y
        return [k, -1., b]    #xzy点斜式参数转换为一般式参数(实际值还为点斜式，a=k，b=-1，c=b)


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        print("平行x轴") if fdConfig.LOG_FOR_EAST_DATA==True else None
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        print("最低点： "+str(poly[p_lowest])) if fdConfig.LOG_FOR_EAST_DATA==True else None
        print("最低点右侧点： "+str(poly[p_lowest_right])) if fdConfig.LOG_FOR_EAST_DATA==True else None
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            print("大于45") if fdConfig.LOG_FOR_EAST_DATA==True else None
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            print("小于45") if fdConfig.LOG_FOR_EAST_DATA==True else None
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def generate_rbox(im_size, polys, tags):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)

    # for poly_idx, poly_tag in enumerate(zip(polys, tags)):
    # poly = poly_tag[0]  #xzy 这里poly是四个元素，每个是一对坐标。
    # tag = poly_tag[1]
    poly_idx = 0
    poly = polys
    tag = tags

    r = [None, None, None, None]
    for i in range(4):
        r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                   np.linalg.norm(poly[i] - poly[(i - 1) % 4]))   #xzy r[]的四个元素，是四个顶点，各自所连的最短边，的边长。  np.linalg.norm(点1-点2)求边长。
    # score map
    shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]    #poly缩小为0.7
    cv2.fillPoly(score_map, shrinked_poly, 1)                     #xzy score mao 完成。
    cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)          #xzt poly_mask 完成
    # if the poly is too small, then ignore it during training
    poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))      #xzy 纵向两个边中选短的，返回其边长
    poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))      #xzy 横向两个边中选短的，返回其边长
    if min(poly_h, poly_w) < FLAGS.min_text_size:                                 #xzy 长宽过于小的poly，是难例，通过training_mask置为0，loss时与score_map抵消，来消除难例。
        cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)                 #xzy training_mask 完成
    if tag:                                                                       #xzy 标签类别是**的，无法识别，是难例，通过training_mask置为0，loss时与score_map抵消，来消除难例。
        cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

    xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))     #xzy xy_in_poly是poly中的点的集合。每个元素是点的x和y坐标组成的数组。
    # if geometry == 'RBOX':
    # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
    fitted_parallelograms = []
    for i in range(4):
        p0 = poly[i]                #xzy 第一次循环：左上点
        p1 = poly[(i + 1) % 4]      #xzy 第一次循环：右上点
        p2 = poly[(i + 2) % 4]      #xzy 第一次循环：右下点
        p3 = poly[(i + 3) % 4]      #xzy 第一次循环：左下点
        #xzy 下面 这一步是多项式拟合，两点拟合一条直线
        edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]]) #xzy   p0[0] 左上点x；  p1[0] 右上点x； p0[1]左上点y；  p1[1]右上点y   第一次循环拟合上横边，返回上横边的直线参数(ax+by+c=0的a、b、c)。
        #xzy edge每次循环，代表一条边。
        backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])  #xzy p0[0] 左上点x；p3[0] 左下点x； p0[1]左上点y； p3[1]左下点y  第一次循环拟合左竖边，返回左竖边的直线参数(ax+by+c=0的a、b、c)。
        forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])   #xzt p1[0] 右上点x；p2[0] 右下点x   p1[1]右上点y； p2[1]右下点y  第一次循环拟合右竖边，返回右竖边的直线参数(ax+by+c=0的a、b、c)。
        if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
            # 平行线经过p2 - parallel lines through p2
            if edge[1] == 0:                    #xzy 参数b=0，上横边与x轴平行
                edge_opposite = [1, 0, -p2[0]]  #xzy 下横边与上横边平行，且经过p2，即参数c为-p2[0]
            else:
                edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]  #xzy
        else:
            # 经过p3 - after p3
            if edge[1] == 0:
                edge_opposite = [1, 0, -p3[0]]
            else:
                edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
        # move forward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p2 = line_cross_point(forward_edge, edge_opposite)
        if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
            # across p0
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p0[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
        else:
            # across p3
            if forward_edge[1] == 0:
                forward_opposite = [1, 0, -p3[0]]
            else:
                forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
        new_p0 = line_cross_point(forward_opposite, edge)
        new_p3 = line_cross_point(forward_opposite, edge_opposite)
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        # or move backward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p3 = line_cross_point(backward_edge, edge_opposite)
        if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
            # across p1
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p1[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
        else:
            # across p2
            if backward_edge[1] == 0:
                backward_opposite = [1, 0, -p2[0]]
            else:
                backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
        new_p1 = line_cross_point(backward_opposite, edge)
        new_p2 = line_cross_point(backward_opposite, edge_opposite)
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
    areas = [Polygon(t).area for t in fitted_parallelograms]
    parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
    # sort thie polygon
    parallelogram_coord_sum = np.sum(parallelogram, axis=1)
    min_coord_idx = np.argmin(parallelogram_coord_sum)
    parallelogram = parallelogram[
        [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

    rectange = rectangle_from_parallelogram(parallelogram)
    rectange, rotate_angle = sort_rectangle(rectange)

    p0_rect, p1_rect, p2_rect, p3_rect = rectange
    for y, x in xy_in_poly:
        point = np.array([x, y], dtype=np.float32)
        # top
        geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
        # right
        geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
        # down
        geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
        # left
        geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
        # angle
        geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask


def generator(input_size=512, batch_size=32,
              vis=False, img=None, text_polys= None, text_tags=None):
    # index = np.arange(0, image_list.shape[0])
    # while True:
    #     np.random.shuffle(index)
    #     images = []
    #     image_fns = []
    #     score_maps = []
    #     geo_maps = []
    #     training_masks = []
    im = img
    print(img.shape) if fdConfig.LOG_FOR_EAST_DATA==True else None
    h, w, _ = im.shape


    #xzy 随机创建背景图片，丢弃。
    # if np.random.rand() < background_ratio:
    #     # crop background
    #     im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
    #     if text_polys.shape[0] > 0:
    #         # cannot find background
    #         continue
    #     # pad and resize image
    #     new_h, new_w, _ = im.shape
    #     max_h_w_i = np.max([new_h, new_w, input_size])
    #     im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    #     im_padded[:new_h, :new_w, :] = im.copy()
    #     im = cv2.resize(im_padded, dsize=(input_size, input_size))
    #     score_map = np.zeros((input_size, input_size), dtype=np.uint8)
    #     geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
    #     geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
    #     training_mask = np.ones((input_size, input_size), dtype=np.uint8)
    # else:

    # xzy 数据增强之随机crop，丢弃。
    # im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
    # h, w, _ = im.shape


    score_map, geo_map, training_mask = generate_rbox((h, w), text_polys, text_tags)

    if vis:
        fig, axs = plt.subplots(3, 2, figsize=(20, 30))
        axs[0, 0].imshow(im[:, :, ::-1])
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        for poly in text_polys:
            poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
            poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
            axs[0, 0].add_artist(Patches.Polygon(
                poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
            axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
        axs[0, 1].imshow(score_map[::, ::])
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[1, 0].imshow(geo_map[::, ::, 0])
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(geo_map[::, ::, 1])
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[2, 0].imshow(geo_map[::, ::, 2])
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
        axs[2, 1].imshow(training_mask[::, ::])
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])
        plt.tight_layout()
        plt.show()
        plt.close()



    # images.append(im[:, :, ::-1].astype(np.float32))
    # image_fns.append(im_fn)
    # score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
    # geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
    # training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
    #
    # if len(images) == batch_size:
    #     yield images, image_fns, score_maps, geo_maps, training_masks
    #     images = []
    #     image_fns = []
    #     score_maps = []
    #     geo_maps = []
    #     training_masks = []


    return score_map, geo_map, training_mask


# ------------------xzy---------------------


'''训练集(手动标注的300图片)'''
# 针对一张图片
def readTrain(txtName):

  pureName = txtName.split('.')[0]
  # print(str(pureName))

  '''读图片'''
  # img = Image.open(fdConfig.arti_img_path + pureName +".jpg")  #读图片
  img = cv2.imread(fdConfig.arti_img_path + pureName +".jpg")
  # img = img.transpose([2, 0, 1])



  # ratio = 2
  # height_resized = fdConfig.IMG_SIZE_HEIGHT
  # width_resized = int(height_resized * ratio)
  # img = img.resize((width_resized, height_resized))
  # img = np.asarray(img)


  '''读标签'''
  stream = open(fdConfig.arti_txt_path + pureName +".txt")   #读标签
  poly = stream.read()
  stream.close()
  # print(poly)
  polyList = poly.split(',')
  polyList = np.asarray(polyList)
  polyList = polyList.astype(np.float)
  print("polyList is: "+str(polyList)) if fdConfig.LOG_FOR_EAST_DATA==True else None


  '''图片和标签联动修改'''

  # 1、pad————图片resize策略，向下填充0像素至正方形，再resize正方形。
      #1>、先pad
  input_size =fdConfig.IMG_SIZE_EAST
  height = img.shape[0]
  width = img.shape[1]
  max_w_h_i = np.max([width,height,input_size])
  print("max_w_h_i is: "+str(max_w_h_i) ) if fdConfig.LOG_FOR_EAST_DATA==True else None
  template = np.zeros([max_w_h_i, max_w_h_i, 3], dtype=np.uint8)
  template[:height, :width, :] = img.copy()
  print("template shape is "+str(template.shape)) if fdConfig.LOG_FOR_EAST_DATA==True else None
  # cv2.imshow("template", template)
  # cv2.waitKey(0)

      #2>、再resize
  print("imput_size is "+str(input_size)) if fdConfig.LOG_FOR_EAST_DATA==True else None
  # img = np.resize(template,[input_size,input_size,3])   #bug! np的resize会破坏图片。
  img = cv2.resize(template,(input_size,input_size))
  print("img shape is "+str(img.shape)) if fdConfig.LOG_FOR_EAST_DATA==True else None
  # cv2.imshow("img", img)
  # cv2.waitKey(0)


      #3>、poly针对resize操作，进行联动修改
  scale_rate = input_size/max_w_h_i
  print(polyList.dtype) if fdConfig.LOG_FOR_EAST_DATA==True else None
  print(scale_rate.dtype) if fdConfig.LOG_FOR_EAST_DATA==True else None
  print("scale_rate is "+str(scale_rate)) if fdConfig.LOG_FOR_EAST_DATA==True else None
  polyList *= float(scale_rate)    #一句代码实现poly联动修改1
  print("resize polylist is: "+str(polyList)) if fdConfig.LOG_FOR_EAST_DATA==True else None
  text_poly = [[ float(polyList[0]), float(polyList[1]) ], [ float(polyList[2]), float(polyList[3]) ],
               [ float(polyList[6]), float(polyList[7]) ], [ float(polyList[4]), float(polyList[5]) ]]
  text_poly = np.asarray(text_poly)
  text_poly = text_poly.astype(np.float)
  text_tag = False if str(polyList[8])=='1' else True
  #由于只有冠字号一个类别，这里全部是“1”。"1"表示可以识别————无模糊字符————不是难例，所以全是False。



  # 2、数据增强————随机resize      xzy因为会导致一个batch中图片不一致，暂时丢弃
  # random_scale = np.array([0.5, 1, 2.0])
  # rd_scale = np.random.choice(random_scale)
  # randomed_size = int(img.shape[0] * rd_scale)
  # randomed_size2 = int(img.shape[1] * rd_scale)
  # img = np.resize(img,(randomed_size  , randomed_size2, 3) )  #这里错误！！np.resize()不可行！！！
  # text_poly *= float(rd_scale)

  # print("random resize polylist is: "+str(text_poly)) if fdConfig.LOG_FOR_EAST_DATA==True else None


  # 3、数据增强————随机crop
  # TODO



  '''生成两个map、一个training_mask'''
  # 调用EAST源码。
  score_map, geo_map, training_mask = generator(input_size=fdConfig.IMG_SIZE_EAST,
                          batch_size=fdConfig.BATCH_SIZE,
                          vis=False,
                          img = img, text_polys=text_poly, text_tags=text_tag)





  score_map = score_map[::4, ::4, np.newaxis]
  geo_map = geo_map[::4, ::4, :]
  training_mask = training_mask[::4, ::4, np.newaxis]

  img = img.transpose([2, 0, 1])
  score_map = score_map.transpose([2, 0, 1])
  geo_map = geo_map.transpose([2, 0, 1])
  training_mask = training_mask.transpose([2, 0, 1])

  return img, score_map, geo_map, training_mask



colorTransform = transforms.ColorJitter()

class FdTrainDataEAST (Data.Dataset):

  def __init__(self):
    txtList = os.listdir(fdConfig.arti_txt_path)
    img_list = []
    score_maps = []
    geo_maps = []
    training_masks = []
    if fdConfig.is_test:
      txtList = txtList[:fdConfig.test_train_num]
    for txtName in txtList:
      img, score_map, geo_map, training_mask = readTrain(txtName)
      print("img shape is : "+str(img.shape)) if fdConfig.LOG_FOR_EAST_DATA==True else None
      print("img type is : "+str(img.dtype)) if fdConfig.LOG_FOR_EAST_DATA==True else None
      img_list.append(img)
      score_maps.append(score_map)
      print("score map shape is : "+str(score_map.shape)) if fdConfig.LOG_FOR_EAST_DATA==True else None
      geo_maps.append(geo_map)
      print("geo_maps shape is : "+str(geo_map.shape)) if fdConfig.LOG_FOR_EAST_DATA==True else None
      training_masks.append(training_mask)
      print("training_masks shape is : "+str(training_mask.shape)) if fdConfig.LOG_FOR_EAST_DATA==True else None


    # score_maps = np.array(score_maps, dtype=np.float32)
    # geo_maps = np.array(geo_maps )
    # training_masks = np.array(training_masks )

    self.x = np.asarray(img_list)
    self.s = np.asarray(score_maps)
    self.g = np.asarray(geo_maps)
    self.m = np.asarray(training_masks)
    self.l = len(self.x)

  def __getitem__(self, index):
    img = self.x[index]
    img = colorTransform(img)
    xtensor = torch.from_numpy(img)
    xFloatTensor = xtensor.type(torch.FloatTensor)

    # ytensor = torch.from_numpy(self.y[index])
    # yFloatTensor = ytensor.type(torch.FloatTensor)

    stensor = torch.from_numpy(self.s[index])
    sFloatTensor = stensor.type(torch.FloatTensor)
    gtensor = torch.from_numpy(self.g[index])
    gFloatTensor = gtensor.type(torch.FloatTensor)
    mtensor = torch.from_numpy(self.m[index])
    mFloatTensor = mtensor.type(torch.FloatTensor)

    return xFloatTensor,sFloatTensor,gFloatTensor,mFloatTensor

  def __len__(self):
    return self.l





'''测试集(比赛中的训练集)'''

def readTest(imgName):

  pureName = imgName.split('.')[0]
  # print(str(pureName))

  '''读图片'''
  img = Image.open(fdConfig.train_img_path + pureName +".jpg")  #读图片
  ratio = 2                           #改用固定长宽（224*112）
  # print("width is "+str(width)+", height is "+str(height),", ratio is "+str(ratio))

  height_resized = fdConfig.IMG_SIZE_HEIGHT
  width_resized = int(height_resized * ratio)


  img = img.resize((width_resized, height_resized))
  img = np.asarray(img)
  img = img.transpose([2, 0, 1])
  return img


class FdTestDataEAST (Data.Dataset):
  def __init__(self):
    trainList = os.listdir(fdConfig.train_img_path)
    img_list = []
    name_list = []
    if fdConfig.is_test:
      trainList = trainList[:fdConfig.test_test_num]
    for imgName in trainList:
      img = readTest(imgName)
      img_list.append(img)

      name_list.append(imgName)

    self.x = np.asarray(img_list)
    self.y = np.asarray(name_list)
    self.l = len(self.x)

  def __getitem__(self, index):
    img = self.x[index]
    # img = colorTransform(img)
    xtensor = torch.from_numpy(img)
    xFloatTensor = xtensor.type(torch.FloatTensor)

    return xFloatTensor,self.y[index]

  def __len__(self):
    return self.l





# if __name__ ==  "__main__":
#   # xml2txt()
#   readTxt()
