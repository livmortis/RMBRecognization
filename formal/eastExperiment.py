import numpy as np
import cv2
import fdConfig
import PIL.Image as Image
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import torch
import os
import heapq

# # fillpoly
# scoremap = np.ones((15,15),np.float32)
# poly = np.array([[5,5],[5,8],[8,5],[8,8]])
# poly2 = np.array([[3,3],[3,9],[9,3],[9,9]])
# cv2.fillPoly(scoremap, [poly,poly2],0.5)
# print(scoremap)
# cv2.imshow('a',scoremap)
# cv2.waitKey(0)


# random size
# img = np.ones((10,10),np.float32)
# poly = np.array([[5,5],[5,8],[8,5],[8,8]])
# # print('before \n'+str(img))
# print('poly before \n'+str(poly))
#
# a = np.random.rand(1)
# print(a)
# img = cv2.resize(img,None,fx=a,fy=a)
# # print('after \n'+str(img))
# poly *= a
# print('poly after \n'+str(poly))


# # size and shape
# path = "../../dataset_formal/detect_data/arti_labeled_img_300/04WQKGDO.jpg"
# a = np.array([[5,5],[5,8],[8,5],[8,8]])
# print("numpy: "+str(a.shape))
# print("numpy: "+str(a.size))
#
# pic = cv2.imread(path)
# print("cv2 :"+str(pic.shape))
# print("cv2 :"+str(pic.size))
#
# pic2 = Image.open(path)
# print("PIL: "+str(pic2.size))


# 多项式拟合
# a = np.arange(1,10,1)
# b = np.array([ 1, 4,8,16 , 27 ,36 ,45 ,60, 81])
# print(a)
# print(b)
# plt.plot(a,b)
# plt.show()
# arg = np.polyfit(a,b,3)
# plt.plot(np.arange(1,30,1), np.polyval(arg , np.arange(1,30,1)))
# plt.show()
# print(arg)
# d = np.polyval(arg, 11)
# print(d)


# 范数
# a = np.array([[5,5],[5,8],[8,5],[8,8]])
# b = np.arange(1,5,1)
# norm2 = np.linalg.norm(b,2)
# norm1 = np.linalg.norm(b,1)
# sqr_sum = sum(b**2)
# print(b)
# print(str(norm2))
# print(str(np.sqrt( sum(np.square(b) ))))




# 点乘和叉乘
# a = np.array([1,2,3])
# b = np.array([2,2,2])
# print(a*b)
# print(np.multiply(a,b))
# print(np.matmul(a,b))
# print(np.dot(a,b))
# print(np.cross(a,b))
# aa = np.array([[5,5],[5,8],[8,5],[8,8]])
# bb = np.array([[2,2],[2,2],[2,2],[2,2]])
# cc = np.array([[2,2,2,2],[2,2,2,2]])
# print("\n")
# print(np.multiply(aa,bb)) #[ [10,10],[10,16],[16,10],[16,16] ]
# print(aa*bb)  #[ [10,10],[10,16],[16,10],[16,16] ]
# print(np.cross(aa,bb))
# print(np.matmul(aa,cc)) # 4X4矩阵
# print(np.dot(aa,cc))  # 4X4矩阵
# print("\n")
# # q= np.array([[2,2],[2,2],[2,2]])
# # w = np.array([[2,2,2,2],[2,2,2,]])
# # print(np.dot(,cc))


# 平行四边形面积
# yes = True
# a = np.array([[5,5],[5,8],[8,8],[8,5]])
# print(Polygon(a).area) if yes==True else None


# np.newaxis测试
# q= np.array([[2,2],[2,2]])
# p = q[np.newaxis, :, :]
# print(q.shape)
# print(q)
# print(p.shape)
# print(p)
# w = q[ :, :,np.newaxis]
# print(w.shape)
# print(w)
# e = q[ :, np.newaxis,:]
# print(e.shape)
# print(e)



# pytorch.Tensor运算测试
# q = np.array([1,2,3])
# w = np.array([2,2,2])
# q = torch.Tensor(q)
# w = torch.Tensor(w)
# print(q)
# print(w)
# a = q.dot(w)
# aa = (q*w).sum()
# print(a)
# print(aa)
# s=q.sum()
# print(s)
# f= np.array([[1,2],[3,4]])
# d= np.array([[2,2],[2,2]])
# f = torch.Tensor(f)
# d = torch.Tensor(d)
# print("二维矩阵点乘结果：" + str(f.dot(d)))    #不可以，因为二维的点积不等于各项相乘再求和，而是遵循矩阵乘法。
# print("二维矩阵各项乘再求和结果：" + str((f*d).sum()))




# pytorch.Tensor运算测试2
# f= np.array([[1,2],[3,4]])
# d= np.array([[2,2],[2,2]])
# f = torch.Tensor(f)
# d = torch.Tensor(d)
# ff = f.split(split_size=1,dim=1)
# # print(ff)
# k = torch.Tensor.min(f,d)
# # print(k)
# # z = f.mean()





# ::符号
#[start : end : step]
# f= np.array([[1,2],[3,4]])
# d = [[1,2],[3,4]]
# print(d)
# print(d[::2])
# a = np.array([1,2,3,4,5,6,7])
# print(a[::-1])    #倒序排列
# print(a[::-2])





# split
# a = np.zeros([1,5,22,22])
# a = torch.Tensor(a)
# print(a.shape)
# q,w,e,r,t = a.chunk(5,1)
# print(q.shape)





# "image resize" and "numpy array resize"
# a = cv2.imread("../../dataset_formal/detect_data/arti_labeled_img_300/08DR6GTY.jpg")
# print(a.shape)
# cv2.imshow("a",a)
# cv2.waitKey(0)
#
# b = cv2.resize(a,(700,300))
# print(b.shape)
# cv2.imshow("b",b)
# cv2.waitKey(0)
#
# c = np.array(b) #c是numpy数组。
# cv2.imshow("c",c)
# cv2.waitKey(0)
#
# d = np.resize(c,[200,600])  #用np.resize()，图片完全被破坏。
# cv2.imshow("d",d)
# cv2.waitKey(0)
#
# e = cv2.resize(c,(600,200,3))
# cv2.imshow("e",e)
# cv2.waitKey(0)
#
# f = e.resize()




# transpose
# a = cv2.imread("../../dataset_formal/detect_data/arti_labeled_img_300/08DR6GTY.jpg")
# b = np.array(a)
# print(b.shape)
# c = b.transpose([2,0,1])
# print(c.shape)
# d = torch.Tensor(c)
# print(d.shape)

# print(e.shape)




# zip
# f= np.array([[1,2],[3,4]])
# g= np.array([[6,6],[8,8]])
# for i,j in zip(f,g):
#   print("begin")
#   print(i)
#   print("")
#   print(j)



# fill_poly可视化测试
# shrinked_poly = np.array([[[231, 159],[295, 159],[295, 165],[231, 165]]])
# score_map = np.zeros((416, 416), dtype=np.uint8)
# cv2.fillPoly(score_map, shrinked_poly, 1)
# cv2.imshow("score_map after fillpoly",score_map)
# cv2.waitKey(0)

# shrinked_poly = np.array([[[4, 4],[8, 4],[8, 8],[4, 8]]])
# score_map = np.zeros((10, 10), dtype=np.uint8)
# cv2.fillPoly(score_map, shrinked_poly, 255)
# cv2.imshow("score_map after fillpoly",score_map)
# cv2.waitKey(0)


# 求余
# a = np.arange(1,20,1)
# for i in range(19):
#   if a[i]%5==0:
#     print(a[i])


# tensor.log测试
# a = torch.Tensor(np.array([[[0.4, 0.4],[0.8, 0.4],[0.8, 0.8],[0.4, 0.8]]]))
# a2 = torch.Tensor(np.array([[[4, 4],[8, 4],[8, 8],[4, 8]]]))
# a3 = -torch.Tensor.log(a+1/a2+1)
# a3 = -torch.Tensor.log((a+1)/(a2+1))
#
# print(a3)


# grep metrics测试
# a = np.array([[3, 119],[2, 159],[4, 165],[231, 109]])
# coordinates = np.argwhere(a>100)
# print(len(coordinates))
# c = []
# for coor in coordinates:
#   print(coor)
#   one = a[coor]
  # print(one)
  # c.append(a[coor])

# print(c)


# read file
# a = cv2.imread("test.jpg")
# print(a)
# list = os.listdir("testfile")
# print(list)


# arg choose
# a = np.array([[3, 119],[12, 59],[4, 165],[231, 109]])
# b = np.sort(a)
# b = np.argwhere(a>10)
# b = np.argmax(a)
# b= heapq.nlargest(3, a.flatten())
# b = a[np.argpartition(-a,1)]
# c = np.argmax(a)
# c = b[np.argsort(b[:, 0])]
# d= c[:, ::-1]
# print(b)

# z = np.array([4,7,2,4,3,7,87,6,4,5,3,2,33,423])
# zz= heapq.nlargest(3, z.flatten())
# print(zz)
# a = np.array([[3, 119],[2, 159],[4, 165],[231, 109]])
# abig10 = a>10
# print(abig10.max(1))

# a = np.array([[3, 119],[12, 59],[4, 165],[231, 109]])
# # a = a>100
# # print(a.max(1))
# print(a.shape)
# for i in a:
#   print(i)
#   print(i.shape)

# a = np.array([[3, 119],[12, 59],[4, 165],[231, 109]])
# a = a>100
# print(a)
# print(a.max(1))

