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
    lrSchedule = Opt.lr_scheduler.ExponentialLR(optm, fdConfig.lr_exponential_gamma)

    dataset_R = fdData_Regression.FdTrainDataReg()
    trainDataloader_R = Dataloader.DataLoader(dataset_R, fdConfig.BATCH_SIZE, shuffle=True )

    for epo in range(fdConfig.EPOCH):
      print("begin "+str(epo)+" epoch")
      for index, (x, y) in tqdm(enumerate(trainDataloader_R, 0)):
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

        print("loss is "+str(loss))

      torch.save(model_R, fdConfig.model_saved+"detect_reg_model.pkl")
      print("model has saved")
      lrSchedule.step()
      print("lr is: " + str(lrSchedule.get_lr()[0]))