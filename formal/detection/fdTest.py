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

def drawRect(pred_batch, y_batch):
  i = 0
  for pred in pred_batch:
    y = y_batch[i]
    img_path = fdConfig.train_img_path +str(y)
    img = cv2.imread(img_path)
    print("minx is : "+str(pred[0]))
    rect = cv2.rectangle(img, (pred[0],pred[1]),(pred[2],pred[3]),color=(0,255,255),thickness=1)
    cv2.imshow("a",rect)
    cv2.waitKey(0)

    i+=1


if __name__ == "__main__":
  if fdConfig.WHICH_MODEL == 'R':
    model_R = torch.load(fdConfig.model_saved + "detect_reg_model.pkl")
    dataset_R = fdData_Regression.FdTestDataReg()
    testDataloader_R = Dataloader.DataLoader(dataset_R, fdConfig.BATCH_SIZE, shuffle=False)

    prediction_list = []
    for index, (x, y) in tqdm(enumerate(testDataloader_R, 0)):
      if fdConfig.use_gpu:
        x = x.cuda()
        model_R = model_R.cuda()
      prediction = model_R(x)
      pred_np = prediction.detach().numpy()
      print(pred_np)
      print(y)
      drawRect(pred_np, y)
      prediction_list.append(pred_np)
