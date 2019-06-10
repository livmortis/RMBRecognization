
import  os
import lmdb
import torch
import numpy as np
import torch
from PIL import Image
import cv2


# a = "../../../../dataset_formal/detect_data/polyImg_Reg-gpu_pad5455/"
# b = os.listdir(a)
# # print(b)
# c = lmdb.open(a, max_readers=1,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False)
#
# print(type(c))

#
# print(numpy.__version__)    #numpy版本： 1.16.0 升级为1.16.4
# print(torch.__version__)
#
#
# # pytorch1.0最新API——CTCLoss参数研究                                                                                #一共16张图片，batchsize为1.
# log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()    #log_probs： 模型输出为 每张图片50*20的矩阵，50个数字的序列，每个数字20种可能标签值。
# print(log_probs.shape)
# targets = torch.randint(1, 20, (16, 30), dtype=torch.long)          #target：标签为 每张图片30个数字
# print(targets.shape)
# input_lengths = torch.full((16,), 50, dtype=torch.long)            #input_lenghts: 表明每张图片输出50个数字
# print("input_lengths的真实值： "+str(input_lengths))
# print(input_lengths.shape)
# print(sum(input_lengths))
#
# target_lengths = torch.randint(10,30,(16,), dtype=torch.long)      #target_lengths：表明每张图片标签为11-28不等长度的数字
# print("target_lengths的真实值： "+str(target_lengths))
# print(target_lengths.shape)
# print(sum(target_lengths))


# # 去除string的‘b
# a = ("b' A0H5466902'", "b' OY82034214'", "b' BN09049018'", "b' PK38902215'", "b' XI72789559'")
# b = []
# for i in a:
#   i = i[3:13]
#   b.append(i)
# b = tuple(b)
# print(b)
# # 错误做法，有'b说明时byte，要用decode变为string。或者先ecode再decode




# squeeze
# a = np.array([[[1,2,3],[3,4,5],[11,22,33],[44,55,66]],[[6,6,6],[7,7,7],[8,8,8],[9,9,9]]])
# print(a.shape)
# q, preds = a.max(2)
# print(preds.shape)
# print(q)
# print(preds)
# preds = preds.squeeze()
# print(preds)

# a = np.array([[[1,2,3],[3,4,5],[11,22,33],[44,55,66]],[[6,6,6],[7,7,7],[8,8,8],[9,9,9]]])
# a = torch.Tensor(a)
# print(a.shape)
# print(a)
# print(a.squeeze(2).shape)
# print(a.squeeze(2))
# q, preds = a.max(2)
# print(preds.shape)
# print(preds)
# preds = preds.squeeze(0)
# print(preds.shape)
# print(preds)


#
# # 图片读取不对
# a = "../../../../dataset_formal/detect_data/polyImg_Reg-gpu_pad5455/CKQIOYET.jpg"
# b = cv2.imread(a)
# # print(b.size)
# # a = np.array([[1,2,1,1],[2,1,1,1]])
# # print(a.shape)
# b = torch.Tensor(b)
# print(b.shape)


# 制作单独验证集
for i in range(10,13):
  print(i)
  print("yes")


































