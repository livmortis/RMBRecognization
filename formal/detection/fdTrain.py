import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as Opt
import torch
import numpy as np
import fdData_Regression
import fdData_EAST
import fdConfig
import fdModel_Regression
import fdModel_EAST
from tqdm import tqdm

def calIou(pred_list , label_list):
  i = 0
  iou_list = []
  for pred in pred_list:
    print("pred is "+ str(pred))
    pred_xmin = pred[0]
    pred_ymin = pred[1]
    pred_xmax = pred[2]
    pred_ymax = pred[3]
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    print("label is "+ str(label_list[i]))
    label_xmin = label_list[i][0]
    label_ymin = label_list[i][1]
    label_xmax = label_list[i][2]
    label_ymax = label_list[i][3]
    label_area = (label_xmax - label_xmin) * (label_ymax - label_ymin)

    xmin_maximum = np.maximum(pred_xmin, label_xmin)
    ymin_maximum = np.maximum(pred_ymin, label_ymin)
    xmax_minimum = np.minimum(pred_xmax, label_xmax)
    ymax_minimum = np.minimum(pred_ymax, label_ymax)



    if xmin_maximum > xmax_minimum or ymin_maximum > ymax_minimum:
      iou = 0
    else:
      intersection = (xmax_minimum - xmin_maximum) * (ymax_minimum - ymin_maximum)
      union = pred_area + label_area - intersection
      iou = intersection / union
      print("iou is " + str(iou))
    iou_list.append(iou)

  return iou_list


if __name__ == "__main__":
  if fdConfig.WHICH_MODEL == 'R':
    if fdConfig.need_load_model:
      model_R = torch.load(fdConfig.model_saved+"detect_reg_model.pkl")
      # model_R = torch.load(fdConfig.model_saved+"detect_reg_model_cpu_loss15_epo8.pkl")
    else:
      model_R = fdModel_Regression.FdModelReg()
    criterion = nn.SmoothL1Loss()
    optm = Opt.Adam(model_R.parameters(),lr=fdConfig.LR,weight_decay=fdConfig.WEIGHT_DECAY)
    # lrSchedule = Opt.lr_scheduler.ExponentialLR(optm, fdConfig.lr_exponential_gamma)  # 学习率下降太快，会使loss下降速度被减缓。而loss初始值太大(226),所以希望前期学习率不要变。
    lrSchedule = Opt.lr_scheduler.ReduceLROnPlateau(optm,'min',fdConfig.lr_shrink_factor,fdConfig.lr_patient,verbose=True)

    dataset_R = fdData_Regression.FdTrainDataReg()
    trainDataloader_R = Dataloader.DataLoader(dataset_R, fdConfig.BATCH_SIZE, shuffle=True )

    for epo in range(fdConfig.EPOCH):
      print("\nbegin "+str(epo)+" epoch")
      pred_list = []
      label_list = []
      for index, (x, y) in enumerate(trainDataloader_R, 0):
        if fdConfig.use_gpu:
          x = x.cuda()
          y = y.cuda()
          model_R = model_R.cuda()
        prediction = model_R(x)
        # print(prediction.detach().numpy())
        # print(y)
        optm.zero_grad()
        loss = criterion(prediction,y)
        loss.backward()
        optm.step()

        if fdConfig.use_gpu:
          loss_np = loss.detach().cpu().numpy()
        else:
          loss_np = loss.detach().numpy()
        print("\nloss is "+str(loss_np)+"\n")

        if fdConfig.use_gpu:
          pred_list.extend(prediction.detach().cpu().numpy())
          label_list.extend(y.detach().cpu().numpy())
        else:
          pred_list.extend(prediction.detach().numpy())
          label_list.extend(y.detach().numpy())
        
      iouList = calIou(pred_list[:fdConfig.train_cal_iou_num], label_list[:fdConfig.train_cal_iou_num])
      print("\n"+iouList)
      iou_average = np.average(np.array(iouList))
      print(str(fdConfig.train_cal_iou_num)+" number sample's iou average is:"+str(iou_average))
      
      torch.save(model_R, fdConfig.model_saved+"detect_reg_model.pkl")
      # print("model has saved")
      lrSchedule.step(loss)
      # print("lr is: " + str(round(lrSchedule.get_lr()[0],2)))   #指数衰减才有该方法，ReduceLROnPlateau改为设置参数verbose=True


  elif fdConfig.WHICH_MODEL == 'E':
    if fdConfig.need_load_model:
      model_E = torch.load(fdConfig.model_saved+"detect_east_model.pkl")
    else:
      model_E = fdModel_EAST.FdModelEast()
    # criterion = nn.SmoothL1Loss()
    optm = Opt.Adam(model_E.parameters(),lr=fdConfig.LR,weight_decay=fdConfig.WEIGHT_DECAY)
    lrSchedule = Opt.lr_scheduler.ReduceLROnPlateau(optm,'min',fdConfig.lr_shrink_factor,fdConfig.lr_patient,verbose=True)

    dataset_E = fdData_EAST.FdTrainDataEAST()
    trainDataloader_E = Dataloader.DataLoader(dataset_E, fdConfig.BATCH_SIZE, shuffle=True )

    loss_E = fdModel_EAST.FdLossEast()

    for epo in range(fdConfig.EPOCH):
      print("\nbegin "+str(epo)+" epoch")
      pred_list = []
      label_list = []
      for index, (x, score_map, geo_map, training_mask) in enumerate(trainDataloader_E, 0):
        if fdConfig.use_gpu:
          x = x.cuda()
          score_map = score_map.cuda()
          geo_map = geo_map.cuda()
          training_mask = training_mask.cuda()
          model_E = model_E.cuda()

        F_score, F_geo = model_E(x)
        optm.zero_grad()

        criterionE =  loss_E(score_map, F_score, geo_map, F_geo, training_mask)
        criterionE.backward()
        optm.step()
        print("loss is " + str(criterionE.detach().numpy()))


      lrSchedule.step(criterionE)
      torch.save(model_E, fdConfig.model_saved+"detect_east_model.pkl")






























