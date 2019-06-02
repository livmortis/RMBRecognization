import torchvision.models as Models
import torch.nn.modules as Modules
import torch
import fdConfig
import numpy as np
import cv2
import PIL.Image as Image

# resnet50 各layer输出层的channel：
# layer1：256  layer2:512  layer3:1024  layer4:2048

# 自定义loss函数
class FdLossEast(Modules.Module):
  def __init__(self):
    super(FdLossEast, self).__init__()
  def forward(self, score_map, F_score, geo_map, F_geo, training_mask):

    # score map loss --- dice loss
    print("score map shape is: "+str(score_map.shape))if fdConfig.LOG_FOR_EAST_LOSS else None
    print("F_score shape is: "+str(F_score.shape))if fdConfig.LOG_FOR_EAST_LOSS else None
    print("geo_map shape is: "+str(geo_map.shape))if fdConfig.LOG_FOR_EAST_LOSS else None
    print("F_geo shape is: "+str(F_geo.shape))if fdConfig.LOG_FOR_EAST_LOSS else None
    print("training_mask shape is: "+str(training_mask.shape))if fdConfig.LOG_FOR_EAST_LOSS else None


    # cv2.imshow('score map',score_map.detach().numpy()[4].transpose([1,2,0]))
    # cv2.waitKey(0)
    # cv2.imshow('F_score',F_score.detach().numpy()[4].transpose([1,2,0]))
    # cv2.waitKey(0)
    # cv2.imshow('training_mask',training_mask.detach().numpy()[4].transpose([1,2,0]))
    # cv2.waitKey(0)

    print("score_map sum is: "+str(score_map.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    print("F_score sum is: "+str(F_score.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    print("geo_map sum is: "+str(geo_map.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    print("F_geo sum is: "+str(F_geo.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    print("training_mask sum is: "+str(training_mask.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None



    # intersection_dice = (score_map * F_score * training_mask).sum()
    intersection_dice = (score_map * F_score ).sum() #暂时去掉training_mask 1
    print("intersection_dice: "+str(intersection_dice)) if fdConfig.LOG_FOR_EAST_LOSS else None
    # union_dice = score_map.dot(training_mask) + F_score.dot(training_mask)  错误！
    # union_dice = (score_map * training_mask).sum() + (F_score * training_mask).sum()
    union_dice = score_map .sum() + F_score .sum() #暂时去掉training_mask 2
    print("union_dice: "+str(union_dice)) if fdConfig.LOG_FOR_EAST_LOSS else None
    dice_loss = 1-(2*intersection_dice/(union_dice+1e-5))
    dice_loss *= 0.01
    print("dice_loss: "+str(dice_loss)) if fdConfig.LOG_FOR_EAST_LOSS else None
    #注：dice_loss是一个数。因为有sum()。


    # geo map loss --- iou loss
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = geo_map.chunk(chunks=5,dim=1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred =F_geo.chunk(chunks=5,dim=1)
    gt_height = d1_gt + d3_gt
    gt_width = d2_gt + d4_gt
    gt_area  = gt_height * gt_width
    pred_height = d1_pred + d3_pred
    pred_width = d2_pred + d4_pred
    pred_area  = pred_height * pred_width
    d1_min = torch.Tensor.min(d1_gt,d1_pred)
    d2_min = torch.Tensor.min(d2_gt,d2_pred)
    d3_min = torch.Tensor.min(d3_gt,d3_pred)
    d4_min = torch.Tensor.min(d4_gt,d4_pred)
    inter_height = d1_min+d3_min
    inter_width = d2_min+d4_min
    intersection_iou = inter_height * inter_width
    print("intersection_iou: "+str(intersection_iou.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    union_iou = gt_area + pred_area - intersection_iou
    print("union_iou: "+str(union_iou.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    iou_loss = -torch.Tensor.log(intersection_iou+1.0 / union_iou+1.0)
    print("iou_loss: "+str(iou_loss.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None

    #注：iou_loss是一个图，尺寸等于标签map


    # angle map loss
    angle_loss = 1 - torch.Tensor.cos(angle_pred-angle_gt)
    print("angle_loss: "+str(angle_loss.sum())) if fdConfig.LOG_FOR_EAST_LOSS else None
    #注：angle_loss是一个图，尺寸等于标签map




    # 总的loss
    geo_loss = iou_loss + 20 * angle_loss
    #注：geo_loss是一个图，尺寸等于标签map
    # loss = (geo_loss * score_map * training_mask).mean()  +   dice_loss
    loss = (geo_loss * score_map ).mean()  +   dice_loss  #暂时去掉training_mask 3
    # print("(geo_loss * score_map * training_mask).mean():  "+str((geo_loss * score_map * training_mask).mean())) if fdConfig.LOG_FOR_EAST_LOSS else None
    print("(geo_loss * score_map ).mean():  "+str((geo_loss * score_map ).mean())) if fdConfig.LOG_FOR_EAST_LOSS else None
    print("loss: "+str(loss)) if fdConfig.LOG_FOR_EAST_LOSS else None
    #注：.mean()之后才变成数，最终loss是一个数。

    return loss



class FdModelEast(Modules.Module):
  def __init__(self):
    super(FdModelEast, self).__init__()

    self.resnet = Models.resnet50(True)
    # self.unpool = torch.nn.MaxUnpool2d(2,2,0)
    #unpool调用时需要参数indice ： - `indices`: the indices given out by `MaxPool2d`, 必须和pool结合使用，弃掉。
    self.unSample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    # 128,64,32是EAST写死的
    self.conv_3 = torch.nn.Conv2d(3072, 128, (1,1)) # 2048+512 = 2560,错！  2048+1024=3072!
    self.conv_2 = torch.nn.Conv2d(640, 64, (1,1))   # 128+512 = 640
    self.conv_1 = torch.nn.Conv2d(320, 32, (1,1))    # 64+256 = 320

    self.conv_score = torch.nn.Conv2d(32,1,(1,1))   #生成score map
    self.sigm = torch.nn.Sigmoid()
    self.conv_geo = torch.nn.Conv2d(32,4,(1,1))   #生成geo map

  def forward(self, input):
    mid = self.resnet.conv1(input)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("first output is: " + str(mid.detach().cpu().numpy().shape))
      else:
        print("first output is: " + str(mid.detach().numpy().shape))

    mid = self.resnet.bn1(mid)
    mid = self.resnet.relu(mid)
    mid = self.resnet.maxpool(mid)
    layer1 = self.resnet.layer1(mid)
    layer2 = self.resnet.layer2(layer1)
    layer3 = self.resnet.layer3(layer2)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("layer3 output is: " + str(layer3.detach().cpu().numpy().shape))
      else:
        print("layer3 output is: " + str(layer3.detach().numpy().shape))
    layer4 = self.resnet.layer4(layer3)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("layer4 output is: " + str(layer4.detach().cpu().numpy().shape))
      else:
        print("layer4 output is: " + str(layer4.detach().numpy().shape))

    # l4_shape = layer4.detach().numpy().shape

    '''第一层concat'''
    # un3 = self.unpool(layer4)    # 必须和pool结合使用，弃掉。
    # un3 = torch.Tensor.resize_(layer4, (l4_shape[2] * 2 , l4_shape[3] * 2))   # cannot resize variables that require grad，弃掉
    un3 = self.unSample(layer4)
    cat3 = torch.cat([layer3,un3],1)   #在第二个维度（共四个）即channel方向进行concat。
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("cat3 shape is: "+str(cat3.detach().cpu().numpy().shape))
      else:
        print("cat3 shape is: "+str(cat3.detach().numpy().shape))
    conv3 = self.conv_3(cat3)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("conv3 shape is: "+str(conv3.detach().cpu().numpy().shape))
      else:
        print("conv3 shape is: "+str(conv3.detach().numpy().shape))

    '''第二层concat'''
    un2 = self.unSample(conv3)
    cat2 = torch.cat([layer2,un2],1)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("cat2 shape is: "+str(cat2.detach().cpu().numpy().shape))
      else:
        print("cat2 shape is: "+str(cat2.detach().numpy().shape))
    conv2 = self.conv_2(cat2)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("conv2 shape is: "+str(conv2.detach().cpu().numpy().shape))
      else:
        print("conv2 shape is: "+str(conv2.detach().numpy().shape))

    '''第三层concat'''
    un1 = self.unSample(conv2)
    cat1 = torch.cat([layer1,un1],1)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("cat1 shape is: "+str(cat1.detach().cpu().numpy().shape))
      else:
        print("cat1 shape is: "+str(cat1.detach().numpy().shape))
    conv1 = self.conv_1(cat1)
    if fdConfig.LOG_FOR_NET_CONSTRUCTION:
      if fdConfig.use_gpu:
        print("conv1 shape is: "+str(conv1.detach().cpu().numpy().shape))   #应该是(-1, 32, 56, 56)  (224*224的情况下)
      else:
        print("conv1 shape is: "+str(conv1.detach().numpy().shape))   #应该是(-1, 32, 56, 56)  (224*224的情况下)

    # 生成score map
    convS = self.conv_score(conv1)
    F_score = self.sigm(convS)

    # 生成geo map
    convG = self.conv_geo(conv1)
    convGSig = self.sigm(convG)
    geo_map = convGSig * fdConfig.IMG_SIZE_EAST
    if fdConfig.LOG_FOR_EAST_MODEL:
      if fdConfig.use_gpu:
        print("pred geo_map shape is: " + str(geo_map.detach().cpu().numpy().shape))
      else:
        print("pred geo_map shape is: " + str(geo_map.detach().numpy().shape))
    # 生成angle map
    conAn = self.conv_score(conv1)
    conAnSig = self.sigm(conAn)
    angle_map = (conAnSig - 0.5) * np.pi/2
    if fdConfig.LOG_FOR_EAST_MODEL:
      if fdConfig.use_gpu:
        print("pred angle_map shape is: " + str(angle_map.detach().cpu().numpy().shape))
      else:
        print("pred angle_map shape is: " + str(angle_map.detach().numpy().shape))


    F_geo = torch.cat((geo_map,angle_map),1)


    return F_score , F_geo

