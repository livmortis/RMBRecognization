from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

import models.crnn as crnn
import time

from tensorboardX import SummaryWriter

writer = SummaryWriter()

'''
需要手动添加的参数：
--cuda
'''

need_load = True
saved_model_path = "expr/netCRNN_99_500.pth"

parser = argparse.ArgumentParser()
# parser.add_argument('--trainRoot', help='path to dataset', default="../../../../dataset_formal/classify_data/crnnData/train_byCTPN_MDB")
# parser.add_argument('--valRoot', help='path to dataset', default="../../../../dataset_formal/classify_data/crnnData/val_byCTPN_MDB")
# 9面值ctpn训练裁剪的poly作为数据集：
parser.add_argument('--trainRoot', help='path to dataset', default="../../../../dataset_formal/classify_data/crnnData/train_9mianzhi_MDB")
parser.add_argument('--valRoot', help='path to dataset', default="../../../../dataset_formal/classify_data/crnnData/val_9mianzhi_MDB")


parser.add_argument('--lr', type=float, default=0.0008, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)



parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
# parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')  #xzy  "the height of conv must be 1"
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
# parser.add_argument('--imgW', type=int, default=288, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
# parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)") # xzy 去掉
# parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--alphabet', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUWXYZ')       #xzy   RMB识别，没有V
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
# parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')    #xzy 错误！！ 当前epoch到不了1000就永远无法保存！！！
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot,type="train")
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    # shuffle=True, sampler=sampler,
    shuffle=True,   #xzy
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)),type="val")

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
# criterion = CTCLoss()
criterion = torch.nn.CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()


if need_load:  #xzy
    print('loading pretrained model from %s' % saved_model_path)
    crnn.load_state_dict(torch.load(saved_model_path))

print(crnn)


image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
# if opt.adam:
if True:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))

    lrSchedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.8,patience=100,verbose=True)

# elif opt.adadelta:
#     optimizer = optim.Adadelta(crnn.parameters())
# else:
#     optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)    #xzy 新pytorch再max(2)之后，已经没有了第2个维度。
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            # if pred == target.lower():
            if pred == target:    #xzy  不转小写，难怪准确率一直0.
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


start = time.time()

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:            #displayInterval=100, 代表100个batch显示一次 （即6400张图片）
            writer.add_scalar("log3/", loss_avg.val(), epoch)

            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))

            avg_time_per_step = int(time.time() - start)   #xzy
            print("during time is: "+ str(avg_time_per_step) +" seconds per epoch")
            start = time.time()
            loss_avg.reset()



        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))

    lrSchedule.step(cost)

writer.close()