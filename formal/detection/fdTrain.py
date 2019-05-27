import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as Opt
import torch
import numpy as np
import fdData_Regression
import fdConfig
import fdModel_Regression
from tqdm import tqdm


if __name__ == "__main__":
  if fdConfig.WHICH_MODEL == 'R':
    if fdConfig.need_load_model:
      model_R = torch.load(fdConfig.model_saved+"detect_reg_model.pkl")
    else:
      model_R = fdModel_Regression.FdModelReg()
    criterion = nn.SmoothL1Loss()
    optm = Opt.Adam(model_R.parameters(),lr=fdConfig.LR,weight_decay=fdConfig.WEIGHT_DECAY)
    # lrSchedule = Opt.lr_scheduler.ExponentialLR(optm, fdConfig.lr_exponential_gamma)  # 学习率下降太快，会使loss下降速度被减缓。而loss初始值太大(226),所以希望前期学习率不要变。
    lrSchedule = Opt.lr_scheduler.ReduceLROnPlateau(optm,'min',0.5,10,verbose=True)

    dataset_R = fdData_Regression.FdTrainDataReg()
    trainDataloader_R = Dataloader.DataLoader(dataset_R, fdConfig.BATCH_SIZE, shuffle=True )

    for epo in range(fdConfig.EPOCH):
      print("\nbegin "+str(epo)+" epoch")
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

        print("loss is "+str(loss_np))

      torch.save(model_R, fdConfig.model_saved+"detect_reg_model.pkl")
      # print("model has saved")
      lrSchedule.step(loss)
      # print("lr is: " + str(round(lrSchedule.get_lr()[0],2)))   #指数衰减才有该方法，ReduceLROnPlateau改为设置参数verbose=True