import torchvision.models as Models
import torch.nn.modules as Modules
import torch
import fdConfig

# resnet50 各layer输出层的channel：
# layer1：256  layer2:512  layer3:1024  layer4:2048

class FdModelReg(Modules.Module):
  def __init__(self):
    super(FdModelReg, self).__init__()

    self.resnet = Models.resnet50(True)
    # self.unpool = torch.nn.MaxUnpool2d(2,2,0)
    #unpool调用时需要参数indice ： - `indices`: the indices given out by `MaxPool2d`, 必须和pool结合使用，弃掉。
    self.unSample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    # 128,64,32是EAST写死的
    self.conv_3 = torch.nn.Conv2d(3072, 128, (1,1)) # 2048+512 = 2560,错！  2048+1024=3072!
    self.conv_2 = torch.nn.Conv2d(640, 64, (1,1))   # 128+512 = 640
    self.conv_1 = torch.nn.Conv2d(320, 32, (1,1))    # 64+256 = 320
    # self.gap = torch.nn.AdaptiveAvgPool2d((0,0))  #云上会报错(pytorch)：RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0]
    self.gap = torch.nn.AdaptiveAvgPool2d((1,1))
    self.linear = torch.nn.Linear(32, 4)
    self.sigm = torch.nn.Sigmoid()

  def forward(self, input):
    mid = self.resnet.conv1(input)
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
    if fdConfig.use_gpu:
      print("layer3 output is: " + str(layer3.detach().cpu().numpy().shape))
    else:
      print("layer3 output is: " + str(layer3.detach().numpy().shape))
    layer4 = self.resnet.layer4(layer3)
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
    if fdConfig.use_gpu:
      print("cat3 shape is: "+str(cat3.detach().cpu().numpy().shape))
    else:
      print("cat3 shape is: "+str(cat3.detach().numpy().shape))
    conv3 = self.conv_3(cat3)
    if fdConfig.use_gpu:
      print("conv3 shape is: "+str(conv3.detach().cpu().numpy().shape))
    else:
      print("conv3 shape is: "+str(conv3.detach().numpy().shape))

    '''第二层concat'''
    un2 = self.unSample(conv3)
    cat2 = torch.cat([layer2,un2],1)
    if fdConfig.use_gpu:
      print("cat2 shape is: "+str(cat2.detach().cpu().numpy().shape))
    else:
      print("cat2 shape is: "+str(cat2.detach().numpy().shape))
    conv2 = self.conv_2(cat2)
    if fdConfig.use_gpu:
      print("conv2 shape is: "+str(conv2.detach().cpu().numpy().shape))
    else:
      print("conv2 shape is: "+str(conv2.detach().numpy().shape))

    '''第三层concat'''
    un1 = self.unSample(conv2)
    cat1 = torch.cat([layer1,un1],1)
    if fdConfig.use_gpu:
      print("cat1 shape is: "+str(cat1.detach().cpu().numpy().shape))
    else:
      print("cat1 shape is: "+str(cat1.detach().numpy().shape))
    conv1 = self.conv_1(cat1)
    if fdConfig.use_gpu:
      print("conv1 shape is: "+str(conv1.detach().cpu().numpy().shape))   #应该是(-1, 32, 56, 56)  (224*224的情况下)
    else:
      print("conv1 shape is: "+str(conv1.detach().numpy().shape))   #应该是(-1, 32, 56, 56)  (224*224的情况下)

    gap = self.gap(conv1)
    if fdConfig.use_gpu:
      print("gap shape is: "+str(gap.detach().cpu().numpy().shape))
    else:
      print("gap shape is: "+str(gap.detach().numpy().shape))

    gap = gap.view(gap.size(0), -1)     # 切记！！！
    if fdConfig.use_gpu:
      print("gap after view shape is: "+str(gap.detach().cpu().numpy().shape))
    else:
      print("gap after view shape is: "+str(gap.detach().numpy().shape))
    liar = self.linear(gap)
    # prediction = self.sigm(liar)


    return liar




# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0

# layer1
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5           [-1, 64, 56, 56]           4,096
#        BatchNorm2d-6           [-1, 64, 56, 56]             128
#               ReLU-7           [-1, 64, 56, 56]               0
#             Conv2d-8           [-1, 64, 56, 56]          36,864
#        BatchNorm2d-9           [-1, 64, 56, 56]             128
#              ReLU-10           [-1, 64, 56, 56]               0
#            Conv2d-11          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-12          [-1, 256, 56, 56]             512
#            Conv2d-13          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-14          [-1, 256, 56, 56]             512
#              ReLU-15          [-1, 256, 56, 56]               0
#        Bottleneck-16          [-1, 256, 56, 56]               0
#            Conv2d-17           [-1, 64, 56, 56]          16,384
#       BatchNorm2d-18           [-1, 64, 56, 56]             128
#              ReLU-19           [-1, 64, 56, 56]               0
#            Conv2d-20           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-21           [-1, 64, 56, 56]             128
#              ReLU-22           [-1, 64, 56, 56]               0
#            Conv2d-23          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-24          [-1, 256, 56, 56]             512
#              ReLU-25          [-1, 256, 56, 56]               0
#        Bottleneck-26          [-1, 256, 56, 56]               0
#            Conv2d-27           [-1, 64, 56, 56]          16,384
#       BatchNorm2d-28           [-1, 64, 56, 56]             128
#              ReLU-29           [-1, 64, 56, 56]               0
#            Conv2d-30           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-31           [-1, 64, 56, 56]             128
#              ReLU-32           [-1, 64, 56, 56]               0
#            Conv2d-33          [-1, 256, 56, 56]          16,384
#       BatchNorm2d-34          [-1, 256, 56, 56]             512
#              ReLU-35          [-1, 256, 56, 56]               0
#        Bottleneck-36          [-1, 256, 56, 56]               0
#            Conv2d-37          [-1, 128, 56, 56]          32,768
#       BatchNorm2d-38          [-1, 128, 56, 56]             256
#              ReLU-39          [-1, 128, 56, 56]               0

# layer2
#            Conv2d-40          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-41          [-1, 128, 28, 28]             256
#              ReLU-42          [-1, 128, 28, 28]               0
#            Conv2d-43          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-44          [-1, 512, 28, 28]           1,024
#            Conv2d-45          [-1, 512, 28, 28]         131,072
#       BatchNorm2d-46          [-1, 512, 28, 28]           1,024
#              ReLU-47          [-1, 512, 28, 28]               0
#        Bottleneck-48          [-1, 512, 28, 28]               0
#            Conv2d-49          [-1, 128, 28, 28]          65,536
#       BatchNorm2d-50          [-1, 128, 28, 28]             256
#              ReLU-51          [-1, 128, 28, 28]               0
#            Conv2d-52          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-53          [-1, 128, 28, 28]             256
#              ReLU-54          [-1, 128, 28, 28]               0
#            Conv2d-55          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-56          [-1, 512, 28, 28]           1,024
#              ReLU-57          [-1, 512, 28, 28]               0
#        Bottleneck-58          [-1, 512, 28, 28]               0
#            Conv2d-59          [-1, 128, 28, 28]          65,536
#       BatchNorm2d-60          [-1, 128, 28, 28]             256
#              ReLU-61          [-1, 128, 28, 28]               0
#            Conv2d-62          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-63          [-1, 128, 28, 28]             256
#              ReLU-64          [-1, 128, 28, 28]               0
#            Conv2d-65          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-66          [-1, 512, 28, 28]           1,024
#              ReLU-67          [-1, 512, 28, 28]               0
#        Bottleneck-68          [-1, 512, 28, 28]               0
#            Conv2d-69          [-1, 128, 28, 28]          65,536
#       BatchNorm2d-70          [-1, 128, 28, 28]             256
#              ReLU-71          [-1, 128, 28, 28]               0
#            Conv2d-72          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-73          [-1, 128, 28, 28]             256
#              ReLU-74          [-1, 128, 28, 28]               0
#            Conv2d-75          [-1, 512, 28, 28]          65,536
#       BatchNorm2d-76          [-1, 512, 28, 28]           1,024
#              ReLU-77          [-1, 512, 28, 28]               0
#        Bottleneck-78          [-1, 512, 28, 28]               0
#            Conv2d-79          [-1, 256, 28, 28]         131,072
#       BatchNorm2d-80          [-1, 256, 28, 28]             512
#              ReLU-81          [-1, 256, 28, 28]               0


# layer3
#            Conv2d-82          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-83          [-1, 256, 14, 14]             512
#              ReLU-84          [-1, 256, 14, 14]               0
#            Conv2d-85         [-1, 1024, 14, 14]         262,144
#       BatchNorm2d-86         [-1, 1024, 14, 14]           2,048
#            Conv2d-87         [-1, 1024, 14, 14]         524,288
#       BatchNorm2d-88         [-1, 1024, 14, 14]           2,048
#              ReLU-89         [-1, 1024, 14, 14]               0
#        Bottleneck-90         [-1, 1024, 14, 14]               0
#            Conv2d-91          [-1, 256, 14, 14]         262,144
#       BatchNorm2d-92          [-1, 256, 14, 14]             512
#              ReLU-93          [-1, 256, 14, 14]               0
#            Conv2d-94          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-95          [-1, 256, 14, 14]             512
#              ReLU-96          [-1, 256, 14, 14]               0
#            Conv2d-97         [-1, 1024, 14, 14]         262,144
#       BatchNorm2d-98         [-1, 1024, 14, 14]           2,048
#              ReLU-99         [-1, 1024, 14, 14]               0
#       Bottleneck-100         [-1, 1024, 14, 14]               0
#           Conv2d-101          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-102          [-1, 256, 14, 14]             512
#             ReLU-103          [-1, 256, 14, 14]               0
#           Conv2d-104          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-105          [-1, 256, 14, 14]             512
#             ReLU-106          [-1, 256, 14, 14]               0
#           Conv2d-107         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-108         [-1, 1024, 14, 14]           2,048
#             ReLU-109         [-1, 1024, 14, 14]               0
#       Bottleneck-110         [-1, 1024, 14, 14]               0
#           Conv2d-111          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-112          [-1, 256, 14, 14]             512
#             ReLU-113          [-1, 256, 14, 14]               0
#           Conv2d-114          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-115          [-1, 256, 14, 14]             512
#             ReLU-116          [-1, 256, 14, 14]               0
#           Conv2d-117         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-118         [-1, 1024, 14, 14]           2,048
#             ReLU-119         [-1, 1024, 14, 14]               0
#       Bottleneck-120         [-1, 1024, 14, 14]               0
#           Conv2d-121          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-122          [-1, 256, 14, 14]             512
#             ReLU-123          [-1, 256, 14, 14]               0
#           Conv2d-124          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-125          [-1, 256, 14, 14]             512
#             ReLU-126          [-1, 256, 14, 14]               0
#           Conv2d-127         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-128         [-1, 1024, 14, 14]           2,048
#             ReLU-129         [-1, 1024, 14, 14]               0
#       Bottleneck-130         [-1, 1024, 14, 14]               0
#           Conv2d-131          [-1, 256, 14, 14]         262,144
#      BatchNorm2d-132          [-1, 256, 14, 14]             512
#             ReLU-133          [-1, 256, 14, 14]               0
#           Conv2d-134          [-1, 256, 14, 14]         589,824
#      BatchNorm2d-135          [-1, 256, 14, 14]             512
#             ReLU-136          [-1, 256, 14, 14]               0
#           Conv2d-137         [-1, 1024, 14, 14]         262,144
#      BatchNorm2d-138         [-1, 1024, 14, 14]           2,048
#             ReLU-139         [-1, 1024, 14, 14]               0
#       Bottleneck-140         [-1, 1024, 14, 14]               0
#           Conv2d-141          [-1, 512, 14, 14]         524,288
#      BatchNorm2d-142          [-1, 512, 14, 14]           1,024
#             ReLU-143          [-1, 512, 14, 14]               0


# layer4
#           Conv2d-144            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-145            [-1, 512, 7, 7]           1,024
#             ReLU-146            [-1, 512, 7, 7]               0
#           Conv2d-147           [-1, 2048, 7, 7]       1,048,576
#      BatchNorm2d-148           [-1, 2048, 7, 7]           4,096
#           Conv2d-149           [-1, 2048, 7, 7]       2,097,152
#      BatchNorm2d-150           [-1, 2048, 7, 7]           4,096
#             ReLU-151           [-1, 2048, 7, 7]               0
#       Bottleneck-152           [-1, 2048, 7, 7]               0
#           Conv2d-153            [-1, 512, 7, 7]       1,048,576
#      BatchNorm2d-154            [-1, 512, 7, 7]           1,024
#             ReLU-155            [-1, 512, 7, 7]               0
#           Conv2d-156            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-157            [-1, 512, 7, 7]           1,024
#             ReLU-158            [-1, 512, 7, 7]               0
#           Conv2d-159           [-1, 2048, 7, 7]       1,048,576
#      BatchNorm2d-160           [-1, 2048, 7, 7]           4,096
#             ReLU-161           [-1, 2048, 7, 7]               0
#       Bottleneck-162           [-1, 2048, 7, 7]               0
#           Conv2d-163            [-1, 512, 7, 7]       1,048,576
#      BatchNorm2d-164            [-1, 512, 7, 7]           1,024
#             ReLU-165            [-1, 512, 7, 7]               0
#           Conv2d-166            [-1, 512, 7, 7]       2,359,296
#      BatchNorm2d-167            [-1, 512, 7, 7]           1,024
#             ReLU-168            [-1, 512, 7, 7]               0
#           Conv2d-169           [-1, 2048, 7, 7]       1,048,576
#      BatchNorm2d-170           [-1, 2048, 7, 7]           4,096
#             ReLU-171           [-1, 2048, 7, 7]               0
#       Bottleneck-172           [-1, 2048, 7, 7]               0


#        AvgPool2d-173           [-1, 2048, 1, 1]               0
#           Linear-174                 [-1, 1000]       2,049,000
# ================================================================


#ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (downsample): Sequential(
#         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#   )
#   (layer2): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#   )
#   (layer3): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (downsample): Sequential(
#         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#   )
#   (layer4): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (downsample): Sequential(
#         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#     )
#   )
#   (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
#   (fc): Linear(in_features=2048, out_features=1000, bias=True)
# )