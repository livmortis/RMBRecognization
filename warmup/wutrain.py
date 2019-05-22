import torch as torch
import torch.utils.data as Data
import torch.optim as Opt
import torch.nn as Nn
import wudata as wudata
import wuconfig as wuconfig
import wumodel as wumodel
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def plotConfuseMatrix(pred, label):
  print("value type number is: " + str(wuconfig.value_num))

  predIndexes = []
  labelIndexes = []
  i = 0

  for onePredFloat in pred:
    predIndex = onePredFloat.argmax()
    if wuconfig.USE_GPU:
      predIndex = predIndex.detach().cpu().numpy()
    else:
      predIndex = predIndex.detach().numpy()

    # predIndex = float(predIndex / wuconfig.value_num)  # 归一化。更新：取消绘制矩形图，改为保存numpy
    predIndexes.append(predIndex)

    labelOne = label[i]
    if wuconfig.USE_GPU:
      labelOne = labelOne.detach().cpu().numpy()
    else:
      labelOne = labelOne.detach().numpy()
    # labelNorm = float(labelOne / wuconfig.value_num)  #归一化。更新：取消绘制矩形图，改为保存numpy
    labelIndexes.append(labelOne)
    i += 1

  cm = confusion_matrix(predIndexes,labelIndexes,labels=[0,1,2,3,4,5,6,7,8])
  # cm =np.fill_diagonal(cm, 0) #归一化后才能让对角为0有意义。  # 归一化。更新：取消绘制矩形图，改为保存numpy
  cm_np = np.asarray(cm)
  np.save(wuconfig.cm_saved_file,cm_np)
  print(cm_np)




def calAccuracy(pred, label, type) :
  totalSum = len(label)
  # softmax = torch.nn.Softmax()
  # predFloat = softmax(pred)       #试试去掉softmax？？
  predFloat = pred
  if type == "batch":
    if  wuconfig.USE_GPU:
      predFloat = predFloat.detach().cpu().numpy()
    else:
      predFloat = predFloat.detach().numpy()  #variable要转换成numpy
  i = 0
  truNum = 0
  for onePredFloat in predFloat:
    # onePredFloat = np.asarray(onePredFloat)
    predIndex = onePredFloat.argmax()
    labelIndex = label[i]

    if predIndex == labelIndex:
      truNum += 1
    else:
      print("\n wrong value image is "+str(i)+" image,label should be "+str(label[i])+", but wrong predict to "+ str(predIndex))

    i += 1
  accuracy = truNum/totalSum
  return accuracy



if __name__ =="__main__":

  trainDataloader = Data.DataLoader(wudata.datasetClass("train"), wuconfig.batch_size, shuffle=True)
  validDataloader = Data.DataLoader(wudata.datasetClass("valid"), wuconfig.batch_size, shuffle=True)


  now_newest_and_saved_model = wuconfig.newest_model_num    #手动更改当前已有的最新模型编号！
  exist_model_path = str(wuconfig.model_saved_path)+"epoch_"+str(now_newest_and_saved_model)+"_saved_model.pkl"
  if(os.path.exists(exist_model_path)):
    model = torch.load(exist_model_path)
    print("\nhas load model: "+"epoch"+str(now_newest_and_saved_model)+"_saved_model.pkl")
    now_newest_and_saved_model += 1
  else:
    model = wumodel.ModelClass()


  optm = Opt.Adam(params=[{'params':model.backbone.fc.parameters(),'lr':wuconfig.lr}],lr=wuconfig.lr*0.1, weight_decay=wuconfig.weight_decay)
  lrSchedule = Opt.lr_scheduler.ExponentialLR(optm, wuconfig.lr_exponential_gamma)
  loss = torch.nn.CrossEntropyLoss()

  print("\nbegin to train")
  for epoch_index in range(wuconfig.epoch):

    # print("\nbegin the "+str(epoch_index)+" epoch train")
    model.train()
    for index, (i,j) in tqdm(enumerate(trainDataloader, 0)):
      # i是x，j是y
      # print("this batch pic is:"+str(i.shape) + "\nthis batch label is:"+ str(j.shape))
      if  wuconfig.USE_GPU:
        model = model.cuda()
        i = i.cuda()
        j = j.cuda()

      pred = model(i)
      optm.zero_grad()
      losstrain = loss(pred, j)
      losstrain.backward()
      optm.step()
      accuracy = calAccuracy(pred, j, "batch")
      print("\nthe "+ str(index)+" batch train loss is " + str(losstrain.detach())+"  accuracy is " + str(accuracy))

    #比如当前训练的模型是18（从配置文件中读取），此次保存为19.pkl，然后变为20，以便下次保存为20.
    will_save_model_path = str(wuconfig.model_saved_path)+"epoch_"+str(now_newest_and_saved_model)+"_saved_model.pkl"
    torch.save(model,will_save_model_path)   #保存模型
    print("\nthe "+str(now_newest_and_saved_model)+" epoch model has saved to "+ will_save_model_path)
    now_newest_and_saved_model += 1

    total_data_pred = []
    total_label = []
    print("\nbegin the "+str(now_newest_and_saved_model)+" epoch valid")
    with torch.no_grad():  # 解决了验证时oom问题
      model.eval()
      for index, (x,y) in tqdm(enumerate(validDataloader, 0)):
        if  wuconfig.USE_GPU:
          model = model.cuda()
          x = x.cuda()
          y = y.cuda()

        pred = model(x)
        lossvalid = loss(pred, y)
        accuracy = calAccuracy(pred, y, "batch")
        print("\nthe "+ str(index)+" batch valid loss is " + str(lossvalid.data)+ " accuracy is " + str(accuracy))
        total_data_pred.extend(pred)
        total_label.extend(y)

    # 一个epoch后，整体验证集数据的准确度
    total_accuracy = calAccuracy(total_data_pred, total_label, "epoch")
    print("\n\nthe epoch "+str(now_newest_and_saved_model)+" total accuracy is "+ str(total_accuracy))

    # 一个epoch后，整体验证集数据的混淆矩阵
    plotConfuseMatrix(total_data_pred, total_label)


    lrSchedule.step()
    print("lr is: "+str(lrSchedule.get_lr()[0]))