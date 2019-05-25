import torch as torch
import torch.nn.modules as nnM
import torchvision.models as tvM
from torchsummary import summary


class ModelClass(nnM.Module):
  def __init__(self):
    super(ModelClass, self).__init__()
    device = torch.device('cpu')
    net = tvM.resnet50()
    summary(net.to(device),(3,224,224))

    self.backbone = tvM.resnet50(pretrained=True)

    # 固定输入尺寸224后才可用的全连接层
    # self.backbone.fc = nnM.Sequential(nnM.Linear(2048, 1024), nnM.Linear(1024, 9))

    #Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)
    # self.backbone.fc = nnM.Sequential(nnM.AdaptiveAvgPool1d(1024), nnM.AdaptiveAvgPool1d(9))


    #Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)
    # self.backbone.fc = nnM.Sequential(nnM.AdaptiveAvgPool1d(9))

    #Expected 3-dimensional tensor, but got 4-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)
    # self.backbone.avgpool = nnM.Sequential( nnM.AdaptiveAvgPool1d(9))

    #Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)  ???
    # self.backbone.avgpool = nnM.AvgPool3d(kernel_size=7, stride=1, padding=0)
    # self.backbone.fc = nnM.AdaptiveAvgPool1d(9)

    # 打印出来resnet后两层观察（224*224输入）：
    #    Bottleneck-172           [-1, 2048, 7, 7]       0
    #    AvgPool2d-173           [-1, 2048, 1, 1]        0
    #    Linear-174                 [-1, 1000]       2,049,000
    # 在AvgPool2d之后，由于池化kernal等于7，已经相当于是全局pool了，输出一定是只有一个像素的2d(channel和batch两个维度)，所以上面一直报错“got 2-dimensional”

    #Expected 3-dimensional tensor, but got 4-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)  ???
    # self.backbone.avgpool = nnM.AdaptiveAvgPool2d((5,5))
    # self.backbone.fc = nnM.AdaptiveAvgPool1d(9)

    #size mismatch, m1: [35 x 51200], m2: [25 x 9] at /Users/soumith/minicondabuild3/conda-bld/pytorch_1524590658547/work/aten/src/TH/generic/THTensorMath.c:2033
    # self.backbone.avgpool = nnM.AdaptiveAvgPool2d((5,5))
    # self.backbone.fc = nnM.Linear(25,9)

    #avgpool output is: (35, 2048, 5, 1)
    #Expected 3-dimensional tensor, but got 4-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)
    # self.backbone.avgpool = nnM.AdaptiveAvgPool2d((5,1))
    # self.backbone.fc = nnM.AdaptiveAvgPool1d(9)

    #avgpool output is: (35,)   说明0尺度还是可以设的
    #Expected 3-dimensional tensor, but got 1-dimensional tensor for argument #1 'self' (while checking arguments for adaptive_avg_pool1d)
    # self.backbone.avgpool = nnM.AdaptiveAvgPool3d((0,5,5))
    # self.backbone.fc = nnM.AdaptiveAvgPool1d(9)

    #avgpool output is: (35, 5, 5)
    #fc output is: (35, 5, 9)     为啥AdaptiveAvgPool1d输出是3d？？？见结论2
    # self.backbone.avgpool = nnM.AdaptiveAvgPool3d((5,5,0))
    # self.backbone.fc = nnM.AdaptiveAvgPool1d(9)

    #avgpool output is: (35, 9, 1, 1)
    #求loss时  RuntimeError: invalid argument 3: only batches of spatial targets supported (3D tensors) but got targets of dimension: 1 at /Users/soumith/minicondabuild3/conda-bld/pytorch_1524590658547/work/aten/src/THNN/generic/SpatialClassNLLCriterion.c:60
    # self.backbone.avgpool = nnM.AdaptiveAvgPool3d((9,1,1))



    #成功一、删掉 fc层，直接一个GAP-3D搞定。
    # self.backbone.avgpool = nnM.AdaptiveAvgPool3d((9,0,0))

    #成功二、留下fc层，用GAP-2D代替原来的pool2d。
    self.backbone.avgpool = nnM.AdaptiveAvgPool2d((1,1))
    self.backbone.fc = nnM.Linear(2048, 9)
    #标准的维度样子：
    # layer4 output is: (35, 2048, 7, 7)
    # avgpool output is: (35, 2048, 1, 1)
    # fc output is: (35, 9)



    #关键bug
    # forward()中重写一遍所有层，居然报错
    # 原因：少了全连接层前的 view() 拉平操作。



    #结论：
    #1、 (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
    #     这一层是肯定要改变的，因为kernel_size为7，已经注定了最后一个feature map必须是7*7.

    #2、AdaptiveAvgPool2d、AdaptiveAvgPool3d，指的是卷积核是2d或3d，不代表输出的维度是多少。    （上面一切的关键问题）
    #     所以AdaptiveAvgPool1d不代表输出就是1维！！！  AdaptiveAvgPool2d和AdaptiveAvgPool3d通过改变参数，才能输出1维

    #3、全连接层前，一定要有一个隐形的view()层。

    #4、 AdaptiveAvgPool2d和AdaptiveAvgPool3d的输入值都可以是4d
    #     而AdaptiveAvgPool1d输入必须是3d，是个坑！因为batchsize、channel就占2d，怎么凑3d都不对称。





  def forward(self, input):
    mid = self.backbone.conv1(input)
    print("first output is: "+str(mid.detach().numpy().shape))
    mid = self.backbone.bn1(mid)
    mid = self.backbone.relu(mid)
    mid = self.backbone.maxpool(mid)
    mid = self.backbone.layer1(mid)
    mid = self.backbone.layer2(mid)
    mid = self.backbone.layer3(mid)
    mid = self.backbone.layer4(mid)
    print("layer4 output is: "+str(mid.detach().numpy().shape))

    mid = self.backbone.avgpool(mid)
    print("avgpool output is: "+str(mid.detach().numpy().shape))

    mid = mid.view(mid.size(0), -1)

    mid= self.backbone.fc(mid)
    print("fc output is: "+str(mid.detach().numpy().shape))


    return mid

    # return self.backbone(input)

if __name__ == "__main__":
  # model = tvM.resnet50(pretrained=True)
  # print(model)
  print("oh")






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