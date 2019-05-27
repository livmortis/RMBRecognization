import pandas as pd
import os
import xml.dom.minidom as mdom
from tqdm import tqdm
import PIL.Image as Image
import fdConfig
import torch.utils.data.dataset as Dataset
import torch
import cv2
import numpy as np

arti_img_path = "../../../dataset_formal/detect_data/arti_labeled_img_300/"
arti_label_path = "../../../dataset_formal/detect_data/arti_labeled_label_300/"
arti_txt_path = "../../../dataset_formal/detect_data/arti_labeled_txt_300/"



def readTxt(txtName):

  pureName = txtName.split('.')[0]

  '''读图片'''
  img = Image.open(arti_img_path + pureName +".jpg")  #读图片
  # print(img.size)
  width = img.size[0]
  height = img.size[1]
  ratio = round(width/height, 2)    #保留两位小数
  # print("width is "+str(width)+", height is "+str(height),", ratio is "+str(ratio))

  height_resized = fdConfig.IMG_SIZE_HEIGHT
  width_resized = int(height_resized * ratio)


  img = img.resize((width_resized, height_resized))
  img = np.asarray(img)
  img = img.transpose([2, 0, 1])


  '''读标签'''
  stream = open(arti_txt_path + pureName +".txt")   #读标签
  quad = stream.read()
  stream.close()
  print(quad)
  quadList = quad.split(',')

  xMin = int(quadList[0])
  xMin_ratio = xMin/width
  xMin_resized = int(width_resized * xMin_ratio)
  yMin = int(quadList[1])
  yMin_ratio = yMin/height
  yMin_resized = int(height_resized * yMin_ratio)
  xMax = int(quadList[4])
  xMax_ratio = xMax/width
  xMax_resized = int(width_resized * xMax_ratio)
  yMax = int(quadList[5])
  yMax_ratio = yMax/height
  yMax_resized = int(height_resized * yMax_ratio)
  xRight_resized = xMax_resized
  yRight_resized = yMin_resized
  xLeft_resized = xMin_resized
  yLeft_resized = yMax_resized
  label = [xMin_resized, yMin_resized,xRight_resized,yRight_resized,xMax_resized,yMax_resized,xLeft_resized, yLeft_resized] #8个坐标

  # print("xMin is "+str(xMin)+", yMin is "+str(yMin),", xMax is "+str(xMax),", yMax is "+str(yMax))
  # print("xMin_resized is "+str(xMin_resized)+", yMin_resized is "+
  #       str(yMin_resized),", xMax_resized is "+str(xMax_resized),", yMax_resized is "+str(yMax_resized))

  #绘制resize后的bbox
  # rect = cv2.rectangle(np.array(img), (xMin_resized,yMin_resized),(xMax_resized,yMax_resized),color=(0,255,255),thickness=1)
  # cv2.imshow("a",rect)
  # cv2.waitKey(0)

  return img, label


class FdTrainDataEast (Dataset):
  def __init__(self):
    txtList = os.listdir(arti_txt_path)
    img_list = []
    label_list = []
    for txtName in txtList[:1]:
      img, label = readTxt(txtName)
      img_list.append(img)
      label_list.append(label)

    self.x = img_list
    self.y = label_list
    self.l = len(self.x)

  def __getitem__(self, index):
      return torch.Tensor(self.x[index]) , torch.Tensor(self.y[index])

  def __len__(self):
      return self.l



if __name__ ==  "__main__":
  readTxt()