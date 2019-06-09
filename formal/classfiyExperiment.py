import numpy as np
import os
import pandas as pd

# # dict字典
# c = {"a":0, "b":1, "c":2}
# print(c.get("b",0))
# a = {}
# b = np.array([10,2,43,114,25,114,64,77,114])
# for i in b:
#   a[i] = a.get(i,0)+1
#   #dict.get()的第二个参数是默认值，暗含了如果没有该key和value就创建它们。
# print(a)
# print(a.keys())   #keys()返回的是“去重的”key组成的数组
# sortedKeys = sorted(a.keys())
# dict_key_to_indict ={key:i for (i,key) in enumerate(sortedKeys) } #创建索引字典的好方法
# print(dict_key_to_indict)
# for citem in c:
#   print(citem)


# # 关于strip
# a = " as dfghjk"
# print(a.strip())


# # 脚本改图片名
# imgpath = "../../dataset_formal/detect_data/polyImg_Reg-gpu_pad5455/"
# imglist = os.listdir(imgpath)
# newList = []
# for name in imglist:
#   newName = name.split('_')[-1]
#   try:
#     os.rename(imgpath+name, imgpath+newName)
#     newList.append(imgpath+newName)
#   except Exception as e:
#     print(str(name)+" is eroor")
#
# print(imglist[:10])
# print(newList[:10])


# # bool与int
# if "":
#   print("1")
#
# if 1:
#   print("2")
# if 0:
#   print("3")
# if "1":
#   print("4")


# # 删掉result.csv中每个name前的0    ---useful script !!!    ----更新：没用！！name前的0是正确的名字。。。
# csv_file = "/Users/xzy/Documents/coder/ML/game/2019/RMB-tinymind/dataset_formal/classify_data/densenClassData/result/test_result_6_7.csv"
# new_csv_file = "/Users/xzy/Documents/coder/ML/game/2019/RMB-tinymind/dataset_formal/classify_data/densenClassData/result/new_test_result_6_7.csv"
# df = pd.read_csv(csv_file)
# namelist = df['name']
# newnamelist = []
# newlabellist = []
# lablelist = df['label']
# k = 0
#
# for item,itemlabel in zip(namelist,lablelist):
#   new_name = str(item)[1:12]
#   newnamelist.append(str(new_name))
#   newlabellist.append(str(itemlabel))
#
# new_df = pd.DataFrame({'name':newnamelist , 'label':lablelist})
# column_order = ['name', 'label']
# new_df = new_df[column_order]
# new_df.to_csv(new_csv_file, index=False)
# # print(new_df.head(10))


# # # 尝试寻找result66.6分原因
# new_csv_file = "/Users/xzy/Documents/coder/ML/game/2019/RMB-tinymind/dataset_formal/classify_data/densenClassData/result/change3_test_result_6_7.csv"
# warm_csv_file = "/Users/xzy/Documents/coder/ML/game/2019/RMB-tinymind/result-final-f2.csv"
# df = pd.read_csv(new_csv_file)
# df2 = pd.read_csv(warm_csv_file)
# print(df.head(10))
# print(df2.head(10))



# csv再试
csv_file = "/Users/xzy/Documents/coder/ML/game/2019/RMB-tinymind/dataset_formal/classify_data/densenClassData/result/test_result_6_7.csv"
df = pd.read_csv(csv_file)
print(df.head())
print(len(df))
i = 0
for i in range(3):
  print("fuck")
  i+=1
print(i)
# print(df['name'][0:5])
# print(df['label'][0:5])







