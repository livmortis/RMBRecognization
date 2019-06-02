import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as Opt
import numpy as np
import fdData_Regression
import fdConfig
import fdModel_Regression
from tqdm import tqdm
import torch
import cv2

import fdData_EAST


def drawRect(pred_batch, y_batch, type):
  i = 0
  for pred in pred_batch:
    y = y_batch[i]
    img_path = fdConfig.train_img_path +str(y)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(fdConfig.IMG_SIZE_HEIGHT * 2,fdConfig.IMG_SIZE_HEIGHT))
    # print("minx is : "+str(pred[0]))
    if type=='Reg':
      adjust_xmin = int(pred[0]-10) #目测box位置，手动调整 (regression模型loss=1.5，结果一般)
      adjust_ymin = int(pred[1]-5)  #目测box位置，手动调整
      adjust_xmax = int(pred[2]+10)  #目测box位置，手动调整
      adjust_ymax = int(pred[3]+10) #目测box位置，手动调整
    else:
      adjust_xmin = pred[0]
      adjust_ymin = pred[1]
      adjust_xmax = pred[2]
      adjust_ymax = pred[3]
    rect = cv2.rectangle(img, (adjust_xmin,adjust_ymin),(adjust_xmax,adjust_ymax),color=(0,255,255),thickness=1)
    cv2.imshow("a",rect)
    cv2.waitKey(0)

    i+=1


if __name__ == "__main__":
  if fdConfig.WHICH_MODEL == 'R':
    model_R = torch.load(fdConfig.model_saved + "detect_reg_model.pkl")
    # model_R = torch.load(fdConfig.model_saved + "detect_reg_model_cpu_loss15_epo8.pkl")
    dataset_R = fdData_Regression.FdTestDataReg()
    testDataloader_R = Dataloader.DataLoader(dataset_R, fdConfig.BATCH_SIZE, shuffle=False)

    prediction_list = []
    for index, (x, y) in tqdm(enumerate(testDataloader_R, 0)):
      if fdConfig.use_gpu:
        x = x.cuda()
        model_R = model_R.cuda()
      prediction = model_R(x)
      if fdConfig.use_gpu:
        pred_np = prediction.detach().cpu().numpy()
      else:
        pred_np = prediction.detach().numpy()
      # print(pred_np)
      # print(y)
      drawRect(pred_np, y, 'Reg')

      prediction_list.append(pred_np)

  elif fdConfig.WHICH_MODEL == 'E':
    model_E = torch.load(fdConfig.model_saved + "detect_east_model.pkl")
    dataset_E = fdData_EAST.FdTestDataEAST()
    testDataloader_E = Dataloader.DataLoader(dataset_E, fdConfig.BATCH_SIZE, shuffle=False)

    prediction_list = []
    for index, (x, y) in tqdm(enumerate(testDataloader_E, 0)):
      if fdConfig.use_gpu:
        x = x.cuda()
        model_E = model_E.cuda()

      F_score, F_geo = model_E(x)























