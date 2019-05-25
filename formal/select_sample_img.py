import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

#function:
#1、check opencv managered img's contour result.
#2、separatly put eight class test img into eight dir, than check the wrong predict.
#3、choose 300 img to artificial label locolization.





train_path = "../../dataset_warm_up/train_data/"
# test_path = "../../dataset_warm_up/public_test_data/"
train_label_file = "../../dataset_warm_up/train_face_value_label.csv"
# test_label_file = "../../result-final.csv"
# exam_path = "../../dataset_formal/expm/"
arti_path = "../../dataset_formal/arti_labeled_img_300/"
TEST =False
LENGTH = 600
WIDTH = 320

df = pd.read_csv(train_label_file)
df = df[:300]   #选300张训练图，手动标注坐标框。
nameList = df['name']
labelList = df[' label']

if TEST:
  nameList = nameList[:5000]


train_pics = []
i = 0
for picName in tqdm(nameList):
  train_one_pic = train_path + picName
  train_cv = cv2.imread(train_one_pic)
  # train_cv = cv2.resize(train_cv,(LENGTH,WIDTH))   #resize

  cv2.imwrite(arti_path+picName, train_cv)    # 放入300张训练图片，作为人工标注对象。

  # labelStr = str(labelList[i])
  # if labelStr == "0.1":
  #   cv2.imwrite(exam_path + "/pic0.1/"+str(i)+".jpg", train_cv)
  # elif  labelStr == "0.2":
  #   cv2.imwrite(exam_path + "/pic0.2/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "0.5":
  #   cv2.imwrite(exam_path + "./pic0.5/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "1.0":
  #   cv2.imwrite(exam_path + "./pic1/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "2.0":
  #   cv2.imwrite(exam_path + "./pic2/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "5.0":
  #   cv2.imwrite(exam_path + "./pic5/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "10.0":
  #   cv2.imwrite(exam_path + "./pic10/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "50.0":
  #   cv2.imwrite(exam_path + "./pic50/" + str(i) + ".jpg", train_cv)
  # elif  labelStr == "100.0":
  #   cv2.imwrite(exam_path + "./pic100/" + str(i) + ".jpg", train_cv)
  # else:
  #   print("wrong")


  i += 1