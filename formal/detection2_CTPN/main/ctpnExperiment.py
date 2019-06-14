
import cv2
import os
import numpy as np

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


def isMaxScore(scores):
  maxIndex = scores.argmax()
  newScores = []
  for i in range(len(scores)):
    if i == maxIndex:
      newScores.append(True)
    else:
      newScores.append(False)
  return newScores


a = np.array([1,2,3,4,5,2,3,4,5,1])
mine = isMaxScore(a)
print(mine)
print(type(mine))
print(np.array(mine))
print(type(np.array(mine)))
his = a>4
print(his)
print(type(his))















