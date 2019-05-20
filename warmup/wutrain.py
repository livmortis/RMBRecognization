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

def calAccuracy(pred, label):
  totalSum = len(label)
  softmax = torch.nn.Softmax()
  predFloat = softmax(pred)

  if  wuconfig.USE_GPU:
    predFloat = predFloat.detach().cpu().numpy()
  else:
    predFloat = predFloat.detach().numpy()  #variable要转换成numpy
  i = 0
  truNum = 0
  for onePredFloat in predFloat:
    # onePredFloat = np.asarray(onePredFloat)
    print("\npred  "+ str(onePredFloat))
    predIndex = onePredFloat.argmax()
    labelIndex = label[i]
    print("\nlabel   "+str(labelIndex))

    if predIndex == labelIndex:
      truNum += 1

    i += 1
  print("\ntruNum is "+str(truNum))
  print("\ntotalSum is "+str(totalSum))
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

    print("\nbegin the "+str(epoch_index)+" epoch train")
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
      print("\nthe "+ str(index)+" batch train loss is " + str(losstrain.data))
      accuracy = calAccuracy(pred, j)
      print("\nthe "+ str(index)+" batch train accuracy is " + str(accuracy))

    #比如当前训练的模型是18（从配置文件中读取），此次保存为19.pkl，然后变为20，以便下次保存为20.
    will_save_model_path = str(wuconfig.model_saved_path)+"epoch_"+str(now_newest_and_saved_model)+"_saved_model.pkl"
    torch.save(model,will_save_model_path)   #保存模型
    print("\nthe "+str(epoch_index)+" epoch model has saved to "+ will_save_model_path)
    now_newest_and_saved_model += 1


    print("\nbegin the "+str(epoch_index)+" epoch valid")
    for index, (x,y) in tqdm(enumerate(validDataloader, 0)):
      if  wuconfig.USE_GPU:
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()

      pred = model(x)
      lossvalid = loss(pred, y)
      print("\nthe "+ str(index)+" batch valid loss is " + str(lossvalid.data))
      accuracy = calAccuracy(pred, y)
      print("\nthe "+ str(index)+" batch valid accuracy is " + str(accuracy))

