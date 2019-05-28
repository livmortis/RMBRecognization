import pandas as pd
import os
import xml.dom.minidom as mdom
from tqdm import tqdm
import PIL.Image as Image
import fdConfig
import torch.utils.data as Data
import torch
import cv2
import numpy as np
import fdConfig
from torchvision import transforms



# 将300张xml标注文件转为txt，运行一次即可。
def xml2txt():
  xmlNameList = os.listdir(fdConfig.arti_label_path)
  try:

    for xmlName in tqdm(xmlNameList):
      # print(xmlName)
      xmlFile = fdConfig.arti_label_path+xmlName
      domTree = mdom.parse(xmlFile)
      annotation = domTree.documentElement
      objList = annotation.getElementsByTagName('object')
      for obj in objList:
        bndboxList = obj.getElementsByTagName('bndbox')
        for bndbox in bndboxList:
          xminList = bndbox.getElementsByTagName('xmin')
          xMin = xminList[0].childNodes[0].data
          yminList = bndbox.getElementsByTagName('ymin')
          yMin = yminList[0].childNodes[0].data
          xmaxList = bndbox.getElementsByTagName('xmax')
          xMax = xmaxList[0].childNodes[0].data
          ymaxList = bndbox.getElementsByTagName('ymax')
          yMax = ymaxList[0].childNodes[0].data
          xRight = xMax
          yRight = yMin
          xLeft = xMin
          yLeft = yMax

          # quad = [int(xMin),int(yMin),int(xRight),int(yRight),int(xMax),int(yMax),int(xLeft),int(yLeft),1]
          pureName = xmlName.split('.')[0]
          # print(pureName)
          stream = open(fdConfig.arti_txt_path + pureName +".txt",'w')
          stream.write(str(xMin))
          stream.write(',')
          stream.write(str(yMin))
          stream.write(',')
          stream.write(str(xRight))
          stream.write(',')
          stream.write(str(yRight))
          stream.write(',')
          stream.write(str(xMax))
          stream.write(',')
          stream.write(str(yMax))
          stream.write(',')
          stream.write(str(xLeft))
          stream.write(',')
          stream.write(str(yLeft))
          stream.write(',')
          stream.write(str(1))
          stream.close()
  except Exception as e:
    print("\nthe "+str(xmlName)+" file parse wrong")
    print(e)


'''训练集'''

def readTxt(txtName):

  pureName = txtName.split('.')[0]
  # print(str(pureName))

  '''读图片'''
  img = Image.open(fdConfig.arti_img_path + pureName +".jpg")  #读图片
  # print(img.size)
  width = img.size[0]
  height = img.size[1]
  # ratio = round(width/height, 2)    #长宽不同的图片无法组成一个batch，要么填充黑色，要么resize成固定长宽
  ratio = 2                           #改用固定长宽（224*112）
  # print("width is "+str(width)+", height is "+str(height),", ratio is "+str(ratio))

  height_resized = fdConfig.IMG_SIZE_HEIGHT
  width_resized = int(height_resized * ratio)


  img = img.resize((width_resized, height_resized))
  img = np.asarray(img)
  img = img.transpose([2, 0, 1])


  '''读标签'''
  stream = open(fdConfig.arti_txt_path + pureName +".txt")   #读标签
  quad = stream.read()
  stream.close()
  # print(quad)
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
  if fdConfig.WHICH_MODEL == 'E':
    label = [xMin_resized, yMin_resized,xRight_resized,yRight_resized,xMax_resized,yMax_resized,xLeft_resized, yLeft_resized] #8个坐标
  elif fdConfig.WHICH_MODEL == 'R':
    label = [xMin_resized, yMin_resized,xMax_resized,yMax_resized]            #4个坐标
  else:
    label = [xMin_resized, yMin_resized,xMax_resized,yMax_resized]            #4个坐标

  # print("xMin is "+str(xMin)+", yMin is "+str(yMin),", xMax is "+str(xMax),", yMax is "+str(yMax))
  # print("xMin_resized is "+str(xMin_resized)+", yMin_resized is "+
  #       str(yMin_resized),", xMax_resized is "+str(xMax_resized),", yMax_resized is "+str(yMax_resized))

  #绘制resize后的bbox
  # rect = cv2.rectangle(np.array(img), (xMin_resized,yMin_resized),(xMax_resized,yMax_resized),color=(0,255,255),thickness=1)
  # cv2.imshow("a",rect)
  # cv2.waitKey(0)

  return img, label



colorTransform = transforms.ColorJitter()

class FdTrainDataReg (Data.Dataset):
  def __init__(self):
    txtList = os.listdir(fdConfig.arti_txt_path)
    img_list = []
    label_list = []
    if fdConfig.is_test:
      txtList = txtList[:fdConfig.test_train_num]
    for txtName in txtList:
      img, label = readTxt(txtName)
      img_list.append(img)
      label_list.append(label)

    self.x = np.asarray(img_list)
    self.y = np.asarray(label_list)
    self.l = len(self.x)

  def __getitem__(self, index):
    img = self.x[index]
    img = colorTransform(img)
    xtensor = torch.from_numpy(img)
    xFloatTensor = xtensor.type(torch.FloatTensor)
    ytensor = torch.from_numpy(self.y[index])
    yFloatTensor = ytensor.type(torch.FloatTensor)

    return xFloatTensor,yFloatTensor

  def __len__(self):
    return self.l





'''测试集'''

def readTrain(imgName):

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


class FdTestDataReg (Data.Dataset):
  def __init__(self):
    trainList = os.listdir(fdConfig.train_img_path)
    img_list = []
    name_list = []
    if fdConfig.is_test:
      trainList = trainList[:fdConfig.test_test_num]
    for imgName in trainList:
      img = readTrain(imgName)
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





if __name__ ==  "__main__":
  # xml2txt()
  readTxt()