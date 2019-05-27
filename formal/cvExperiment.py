import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
# print(str(cv2.__version__))   #3.4.0


LENGTH = 600
WIDTH = 320
train_path = "../../dataset_warm_up/train_data/"
label_file = "../../dataset_warm_up/train_face_value_label.csv"
# 满分9分

# 灰度 + 直方图 + 高斯 + 反向二值 + 闭运算 = 8分
# 灰度 + 直方图 + 高斯 + 反向二值 + 先闭后开 = 8分        成功了4669张, 共10000张
# 灰度 + 直方图 + 高斯 + 反向二值 + 开运算 = 8分(实为7分)     但五毛是错的.
# 灰度 + 直方图 + 高斯 + 反向二值 + 先开后闭 = 7分
# 灰度 + 直方图 + 高斯 + 反向二值         = 8分
# 灰度 +         高斯 + 反向二值 + 闭运算 = 6分     一块和十块(特点：非常暗)搞不定，
# 灰度 + 直方图       + 反向二值 + 闭运算 = 4分
# 灰度 + 直方图       + 反向二值 +        = 5分
# 灰度 + 直方图 + 高斯 + 反向二值 + 闭运算 + canny = 0分   效果不好，每张会多出其他小轮廓

#结论
#1、 闭运算白色多，开运算黑色多。开运算检测出的轮廓也更多，也会导致大轮廓拆分成许多小轮廓，导致无法获得外围大轮廓。



a0 = cv2.imread("../../dataset_formal/expm/samp0.jpg")
a1 = cv2.imread("../../dataset_formal/expm/samp1.jpg")
a2 = cv2.imread("../../dataset_formal/expm/samp2.jpg")
a3 = cv2.imread("../../dataset_formal/expm/samp3.jpg")
a4 = cv2.imread("../../dataset_formal/expm/samp4.jpg")
a5 = cv2.imread("../../dataset_formal/expm/samp5.jpg")
a6 = cv2.imread("../../dataset_formal/expm/samp6.jpg")
a7 = cv2.imread("../../dataset_formal/expm/samp7.jpg")
a8 = cv2.imread("../../dataset_formal/expm/samp8.jpg")
s = cv2.imread("../../dataset_formal/expm/s.jpg")
aSum=[a0,a1,a2,a3,a4,a5,a6,a7,a8]
# a = a0

successNum = 0
totalNum = 0
PREVIEW_EVERY_STEP = True
TEST = True

# 真数据实验
# df = pd.read_csv(label_file)
# nameList = df['name']
# if TEST:
#   nameList = nameList[:10000]
# train_pics = []
# for picName in tqdm(nameList):
#   train_one_pic = train_path+picName
#   train_cv = cv2.imread(train_one_pic)
#   train_pics.append(train_cv)


for a in aSum:              # 测试少量数据
# for a in tqdm(train_pics):      # 真实大量数据

  aa = cv2.resize(a,(LENGTH,WIDTH))   #resize

  # (b, r, g) = cv2.split(aa)               ########彩色直方图均衡化
  # b = cv2.equalizeHist(b)
  # cv2.imshow("x", b)
  # cv2.waitKey(0)
  # r = cv2.equalizeHist(r)
  # cv2.imshow("x", r)
  # cv2.waitKey(0)
  # g = cv2.equalizeHist(g)
  # cv2.imshow("x", g)
  # cv2.waitKey(0)
  # cv2.imshow("x", cv2.merge((b,r,g)))
  # cv2.waitKey(0)
  # a = cv2.merge((b,r,g))
  # cv2.imshow("color hist and gray", cv2.cvtColor(cv2.merge((b,r,g)),cv2.COLOR_BGR2GRAY))
  # cv2.waitKey(0)



  '''灰度化————————必须！！！'''
  a = cv2.cvtColor(aa, cv2.COLOR_RGB2GRAY)
  if PREVIEW_EVERY_STEP:
    cv2.imshow("1_gray",a)
    cv2.moveWindow("1_gray",0,0)
    cv2.waitKey(0)

  '''灰度直方图均衡化'''
  a = cv2.equalizeHist(a)
  if PREVIEW_EVERY_STEP:
    cv2.imshow("2_hist",a)
    cv2.moveWindow("2_hist",100,10)
    cv2.waitKey(0)

  '''高斯滤波 ———————— 五星*****'''
  a = cv2.blur(a,(3,3),10)
  if PREVIEW_EVERY_STEP:
    cv2.imshow("3_guassion",a)
    cv2.moveWindow("3_guassion",100,450)
    cv2.waitKey(0)

  '''二值化——————轮廓检测必须项！！！'''
  ret, a = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY_INV)
  # ret, a = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
  if PREVIEW_EVERY_STEP:
    cv2.imshow("4_thresh",a)
    cv2.moveWindow("4_thresh",800,450)
    cv2.waitKey(0)

  '''边缘检测，抛弃'''
  # a = cv2.Canny(a,100,200,edges=None,apertureSize=3,L2gradient=False)
  # if PREVIEW_EVERY_STEP:
  #   cv2.imshow("5_canny",a)
  #   cv2.moveWindow("5_canny",800,350)
  #   cv2.waitKey(0)

  '''形态学滤波'''
  kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
  # a = cv2.erode(a,kernal )                                        ########形态学滤波——腐蚀
  a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernal)             ########形态学滤波——闭运算
  # a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernal)             ########形态学滤波——开运算
  if PREVIEW_EVERY_STEP:
    cv2.imshow("6_close", a)
    cv2.moveWindow("6_close", 800, 10)
    cv2.waitKey(0)


  '''轮廓检测'''
  image, contours, hierarchy = cv2.findContours(a,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(a, contours, -1, (135, 210, 255), 5)     #参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
  if PREVIEW_EVERY_STEP:
    cv2.imshow("7_contour", a)
    cv2.moveWindow("7_contour", 300, 10)
    cv2.waitKey(0)


  for cont in contours:
    '''轮廓外接矩形'''
    rect = cv2.minAreaRect(cont)
    rectWith = rect[1][0]
    rectLen = rect[1][1]

    if rectWith > WIDTH*0.5 and rectLen > LENGTH*0.5 and rectWith< WIDTH*0.9:
      # print("rectWith: " + str(rectWith))
      # print("\nWIDTH: " + str(WIDTH))
      # print("\nrectLen: " + str(rectLen))
      # print("\nLENGTH: " + str(LENGTH))

      box = cv2.boxPoints(rect)
      box = np.int0(box)

      for i in range(len(box)):
        d = cv2.line(aa, tuple(box[i]), tuple(box[(i+1)%4]), (0,255,255),5,2)
      if PREVIEW_EVERY_STEP:
        cv2.imshow("8_rect"+str(successNum), d)
        cv2.moveWindow("8_rect"+str(successNum), 300, 500)
        cv2.waitKey(0)

      successNum +=1

  totalNum +=1

print("成功了"+str(successNum)+"张, 共" +str(totalNum)+"张")














# 知识点


#     cv2.RETR_EXTERNAL表示只检测外轮廓
#     cv2.RETR_LIST检测的轮廓不建立等级关系
#     cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
#     cv2.RETR_TREE建立一个等级树结构的轮廓。

#     cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
#     cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
#     cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法

 # getStructuringElement()时的形状
 # cv2.MORPH_RECT  # 矩形结构
 # cv2.MORPH_ELLIPSE   # 椭圆结构
 # cv2.MORPH_CROSS   # 十字形结构