import numpy as np


# dict字典
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


# 关于strip
a = " as dfghjk"
print(a.strip())