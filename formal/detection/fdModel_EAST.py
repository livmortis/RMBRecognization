import torchvision.models as Models
import torch.nn.modules as Modules
import torch

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
    self.gap = torch.nn.AdaptiveAvgPool2d((0,0))
    self.linear = torch.nn.Linear(32, 4)
    self.sigm = torch.nn.Sigmoid()

  def forward(self, input):
    mid = self.resnet.conv1(input)
    print("first output is: " + str(mid.detach().numpy().shape))
    mid = self.resnet.bn1(mid)
    mid = self.resnet.relu(mid)
    mid = self.resnet.maxpool(mid)
    layer1 = self.resnet.layer1(mid)
    layer2 = self.resnet.layer2(layer1)
    layer3 = self.resnet.layer3(layer2)
    print("layer3 output is: " + str(layer3.detach().numpy().shape))
    layer4 = self.resnet.layer4(layer3)
    print("layer4 output is: " + str(layer4.detach().numpy().shape))

    l4_shape = layer4.detach().numpy().shape

    '''第一层concat'''
    # un3 = self.unpool(layer4)    # 必须和pool结合使用，弃掉。
    # un3 = torch.Tensor.resize_(layer4, (l4_shape[2] * 2 , l4_shape[3] * 2))   # cannot resize variables that require grad，弃掉
    un3 = self.unSample(layer4)
    cat3 = torch.cat([layer3,un3],1)   #在第二个维度（共四个）即channel方向进行concat。
    print("cat3 shape is: "+str(cat3.detach().numpy().shape))
    conv3 = self.conv_3(cat3)
    print("conv3 shape is: "+str(conv3.detach().numpy().shape))

    '''第二层concat'''
    un2 = self.unSample(conv3)
    cat2 = torch.cat([layer2,un2],1)
    print("cat2 shape is: "+str(cat2.detach().numpy().shape))
    conv2 = self.conv_2(cat2)
    print("conv2 shape is: "+str(conv2.detach().numpy().shape))

    '''第三层concat'''
    un1 = self.unSample(conv2)
    cat1 = torch.cat([layer1,un1],1)
    print("cat1 shape is: "+str(cat1.detach().numpy().shape))
    conv1 = self.conv_1(cat1)
    print("conv1 shape is: "+str(conv1.detach().numpy().shape))   #应该是(-1, 32, 56, 56)  (224*224的情况下)

    gap = self.gap(conv1)
    liar = self.linear(gap)
    prediction = self.sigm(liar)


    return prediction
