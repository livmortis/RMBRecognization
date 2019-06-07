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
import time
# import lanms
import matplotlib.pyplot as plt
import os

def writePred(prediction_list, img_name_list):
#  rightLen = 39620 if not fdConfig.is_test else fdConfig.test_test_num

#  if (len(prediction_list) != rightLen):
#    print("length error!： "+str(len(prediction_list)))
#    raise RuntimeError('length error!')


  for (pre, name) in zip(prediction_list, img_name_list):
    pureName = str(name).split(".")[0]
    # print(pureName)
    stream = open(fdConfig.output_reg_path + pureName + ".txt", 'w')
    stream.write(str(pre[0]))
    stream.write(',')
    stream.write(str(pre[1]))
    stream.write(',')
    stream.write(str(pre[2]))
    stream.write(',')
    stream.write(str(pre[3]))
    stream.close()
  print("done")



def drawRect(pred_batch, y_batch, type):
  i = 0
  for pred in pred_batch:
    y = y_batch[i]
    img_path = fdConfig.train_img_path +str(y)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(fdConfig.IMG_SIZE_HEIGHT * 2,fdConfig.IMG_SIZE_HEIGHT))
    # print("minx is : "+str(pred[0]))
    if type=='Reg':
      adjust_xmin = int(pred[0]-5) #目测box位置，手动调整 (regression模型loss=1.5，结果一般)
      adjust_ymin = int(pred[1]-5)  #目测box位置，手动调整
      adjust_xmax = int(pred[2]+5)  #目测box位置，手动调整
      adjust_ymax = int(pred[3]+5) #目测box位置，手动调整
    else:
      adjust_xmin = pred[0]
      adjust_ymin = pred[1]
      adjust_xmax = pred[2]
      adjust_ymax = pred[3]
    rect = cv2.rectangle(img, (adjust_xmin,adjust_ymin),(adjust_xmax,adjust_ymax),color=(0,255,255),thickness=1)
    cv2.imshow("a",rect)
    cv2.waitKey(0)


    i+=1







def detect(score_map, geo_map, timer, score_map_thresh=fdConfig.east_detect_scoremap_thresh, box_thresh=0.1, nms_thres=0.2):
  '''
  restore text boxes from score map and geo map
  :param score_map:
  :param geo_map:
  :param timer:
  :param score_map_thresh: threshhold for score map
  :param box_thresh: threshhold for boxes
  :param nms_thres: threshold for nms
  :return:
  '''
  if len(score_map.shape) == 4:
    score_map = score_map[0, :, :, 0]
    geo_map = geo_map[0, :, :, ]
  # filter the score map
  print("score map shape is: "+str(score_map.shape)) if fdConfig.LOG_FOR_EAST_TEST else None
  xy_text = np.argwhere(score_map > score_map_thresh)
  print("score map shape after thresh is: "+str(score_map.shape)) if fdConfig.LOG_FOR_EAST_TEST else None

  # sort the text boxes via the y axis
  print("xy_text is: "+str(xy_text)) if fdConfig.LOG_FOR_EAST_TEST else None  #空
  xy_text = xy_text[np.argsort(xy_text[:, 0])]
  # restore
  start = time.time()
  text_box_restored = fdData_EAST.restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
  print("text_box_restored is: "+str(text_box_restored)) if fdConfig.LOG_FOR_EAST_TEST else None  #空

  print('{} text boxes before nms'.format(text_box_restored.shape[0]))
  boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
  boxes[:, :8] = text_box_restored.reshape((-1, 8))
  boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
  timer['restore'] = time.time() - start
  # nms part
  start = time.time()
  # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
  # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres) #xzy 临时去掉NMS
  timer['nms'] = time.time() - start

  if boxes.shape[0] == 0:
    return None, timer

  # here we filter some low score boxes by the average score map, this is different from the orginal paper
  for i, box in enumerate(boxes):
    mask = np.zeros_like(score_map, dtype=np.uint8)
    cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
    boxes[i, 8] = cv2.mean(score_map, mask)[0]
  boxes = boxes[boxes[:, 8] > box_thresh]

  return boxes, timer







if __name__ == "__main__":
  if fdConfig.WHICH_MODEL == 'R':
    model_R = torch.load(fdConfig.model_saved + fdConfig.MODEL_NAME)
    if fdConfig.detect_poly_of_train_or_test == "detect_train":
      dataset_R = fdData_Regression.FdTestDataReg()     # 对39620个训练集进行预测poly框
    else:     # ==“detect_test”
      dataset_R = fdData_Regression.FdTestTestDataReg()     # 对20000个测试集进行预测poly框


    testDataloader_R = Dataloader.DataLoader(dataset_R, fdConfig.BATCH_SIZE, shuffle=False)

    with torch.no_grad():  # 解决了测试时oom问题
      model_R.eval()

      prediction_list = []
      img_name_list = []
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
        # drawRect(pred_np, y, 'Reg')   #预览pre框

        prediction_list.extend(pred_np)
        img_name_list.extend(y)
        print(str(index)+" batch")

    writePred(prediction_list, img_name_list)






  elif fdConfig.WHICH_MODEL == 'E':
    # model_E = torch.load(fdConfig.model_saved + "detect_east_model.pkl")
    model_E = torch.load(fdConfig.model_saved + fdConfig.MODEL_NAME)
    dataset_E = fdData_EAST.FdTestDataEAST()
    testDataloader_E = Dataloader.DataLoader(dataset_E, fdConfig.BATCH_SIZE, shuffle=False)

    prediction_list = []
    for index, (x, y) in tqdm(enumerate(testDataloader_E, 0)):
      start_time = time.time()

      if fdConfig.use_gpu:
        x = x.cuda()
        model_E = model_E.cuda()

      timer = {'net': 0, 'restore': 0, 'nms': 0}
      start = time.time()

      # cv2.imshow("in dataloader x" , x[0].detach().numpy().transpose([1, 2, 0]))
      # cv2.waitKey(0)
      # plt.imshow(x[0].detach().numpy().transpose([1, 2, 0]))
      # plt.show()
      F_score, F_geo = model_E(x)

      # print(F_score.shape)
      # print(F_score.sum())
      # print(F_score>0,8)
      # print(F_score)
      # print(F_geo)
      # cv2.imshow("in dataloader F_score" , F_score[0].detach().numpy().transpose([1, 2, 0]))
      # cv2.waitKey(0)
      # cv2.imshow("in dataloader F_geo" , F_geo[0][0].detach().numpy())
      # cv2.waitKey(0)
      if fdConfig.use_gpu:
        F_score = F_score.detach().cpu().numpy()
      else:
        F_score = F_score.detach().numpy()

      coord = np.argwhere(F_score>0.8)
      print("the length of bigger than 0.8 in F_score is: "+str(len(coord)))

      print("F_score shape is : " +str(F_score.shape)) if fdConfig.LOG_FOR_EAST_TEST else None
      F_score = F_score.transpose([0,2,3,1])
      print("F_score shape after transpose is : " +str(F_score.shape)) if fdConfig.LOG_FOR_EAST_TEST else None
      if fdConfig.use_gpu:
        F_geo = F_geo.detach().cpu().numpy()
      else:
        F_geo = F_geo.detach().numpy()
      print("F_geo shape is : " +str(F_geo.shape)) if fdConfig.LOG_FOR_EAST_TEST else None
      F_geo = F_geo.transpose([0,2,3,1])
      print("F_geo shape after transpose is : " +str(F_geo.shape)) if fdConfig.LOG_FOR_EAST_TEST else None


      for each_F_score, each_F_geo ,each_x in zip(F_score,F_geo,x):
        # cv2.imshow("each_F_score" , each_F_score[:,:,0])
        # cv2.waitKey(0)
        # cv2.imshow("each_F_geo" , each_F_geo[:,:,0])
        # cv2.waitKey(0)
        each_F_score = each_F_score[np.newaxis,:,:,:]
        each_F_geo = each_F_geo[np.newaxis,:,:,:]
        print("each_F_score shape is : " + str(each_F_score.shape)) if fdConfig.LOG_FOR_EAST_TEST else None
        print("each_F_geo shape is : " + str(each_F_geo.shape)) if fdConfig.LOG_FOR_EAST_TEST else None

        timer['net'] = time.time() - start
        boxes, timer = detect(score_map=each_F_score, geo_map=each_F_geo, timer=timer)
        # print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(x, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
        duration = time.time() - start_time
        # print('[timing] {}'.format(duration))


        print("boxes is : " +str(boxes)) if fdConfig.LOG_FOR_EAST_TEST else None
        print("image shape is : " +str(each_x.shape)) if fdConfig.LOG_FOR_EAST_TEST else None





















