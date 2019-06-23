
import cv2
import os
import numpy as np
import shutil
import PIL.Image as Image

# # 寻找正确数据路径
# path = "../../../../dataset_formal/detect_data/CTPNData/image"
# pic = "../../../../../dataset_formal/detect_data/CTPNData/image/0A2PDULI.jpg"
# list = os.listdir(path)
# print(list[0])
#
# a = cv2.imread(path)
# print(a)
# # cv2.imshow("a",a)
# # cv2.waitKey(0)



# np.where实验


# def isMaxScore(scores):
#   maxIndex = scores.argmax()
#   newScores = []
#   for i in range(len(scores)):
#     if i == maxIndex:
#       newScores.append(True)
#     else:
#       newScores.append(False)
#   return newScores
#
#
# a = np.array([1,2,3,4,5,2,3,4,5,1])
# mine = isMaxScore(a)
# print(mine)
# print(type(mine))
# print(np.array(mine))
# print(type(np.array(mine)))
# his = a>4
# print(his)
# print(type(his))



# # np.save实验
# npy_path = '../../../../dataset_formal/detect_data/CTPNData/'
# cv2.imwrite(npy_path+"hahahahah", np.zeros([100, 100, 3]))
#
# # bbox_pred_val_list = np.array([88,33,11])
# # np.save(npy_path+"666", bbox_pred_val_list)
# # a = np.load(npy_path+"666.npy")
# # print(a)



# print(str(os.path.basename("../../../dataset_warm_up/train_data/8LEV6DXN.jpg")))




# # for in测试
# a = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# for i in a[0:3]:
#   print(i)
# for j in a[3:6]:
#   print(j)
# for l in a[6:15]:
#   print(l)
#
# print(os.path.join('1','2','3'))
# im_fn =' ../../../dataset_warm_up/train_data/0BRO7XVG.jpg'
# print("xzyxzyxzy"+str(os.path.basename(im_fn)))








'''useful脚本——————ctpn删除broken图片'''
# train1 = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_train_1'
# train2 = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_train_2'
# train3 = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_train_3'
# train4 = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_train_4'
# origin = '../../../../dataset_formal/detect_data/lab_del_waxzy/artificial_crop_broken_img'
# traindataset = '../../../../dataset_warm_up/train_data/'

'''
# 1、从train1中找到xzywa图，截取真实图片名；
# 2、从traindataset中找到该真实图片；
# 3、将其复制到artificial_crop_broken_img文件夹中。
# 4、创建最终的polyImg_CTPN_train文件
# 5、artificial_crop_broken_img文件夹中图片都手动切割，切好的图片也放到polyImg_CTPN_train中 （此时307张）
# 6、polyImg_CTPN_train1————4都放到polyImg_CTPN_train中  （此时39620+307=39927）
# 7、polyImg_CTPN_train中删掉多余的xzywa图片。清点数目--39620.
'''

# train1List = os.listdir(train1)
# t1Num = 0
# listT1 = []
# for t1 in train1List:
#   if len(t1) > 13:
#     t1Num += 1
#     t1 = str(t1)
#     t1_new = t1[5:17]
#     listT1.append(t1_new)
#
# success_cp = 0
# for t1new in listT1:
#   shutil.copy(traindataset + t1new, origin)
#   success_cp += 1
#
# print(success_cp)
# print(t1Num)


# test1 = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_test_1'
# test2 = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_test_2'
# origin_test = '../../../../dataset_formal/detect_data/lab_del_waxzy/origin_img_test'
# testdataset = '../../../../dataset_warm_up/public_test_data/'

#
#
# test2List = os.listdir(test2)
# t2Num = 0
# listT2 = []
# for t2 in test2List:
#   if len(t2) > 13:
#     t2Num += 1
#     t2 = str(t2)
#     t2_new = t2[5:17]
#     listT2.append(t2_new)
#
# success_cp = 0
# for t2new in listT2:
#   shutil.copy(testdataset + t2new, origin_test)
#   success_cp += 1
#
# print(success_cp)
# print(t2Num)


# final_train_path = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_train/'
# tmp = '../../../../dataset_formal/detect_data/lab_del_waxzy/tmp'
# final_list = os.listdir(final_train_path)
# i = 0
# for final in final_list:
#   if len(final) > 13:
#     i += 1
#     # shutil.move(final_train_path+final, tmp)
#
# print(i)


# final_test_path = '../../../../dataset_formal/detect_data/lab_del_waxzy/polyImg_CTPN_test/'
# tmp = '../../../../dataset_formal/detect_data/lab_del_waxzy/tmp'
# final_list = os.listdir(final_test_path)
# i = 0
# for final in final_list:
#   if len(final) > 13:
#     i += 1
#     shutil.move(final_test_path+final, tmp)
#
# print(i)
'''useful脚本——————ctpn删除broken图片    结束'''






















































