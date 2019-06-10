
import torch
import torch.nn
import dataset
import utils
import models.crnn as crnn
from torch.autograd import Variable
import pandas as pd
import numpy as np

saved_model_path = "expr/netCRNN_24_500.pth"




alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUWXYZ'

use_cuda = True



def test_by_xzy(net, test_dataset):
  print('Start test')


  for p in net.parameters():
      p.requires_grad = False

  net.eval()
  data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=True, batch_size=64, num_workers=int(2))
  val_iter = iter(data_loader)

  img_name_List = []
  img_pred_List = []

  for i in range(len(data_loader)):
    data = val_iter.next()
    i += 1
    cpu_images, cpu_img_name = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_img_name)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = net(image)


    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    _, preds = preds.max(2)
    # preds = preds.squeeze(2)    #xzy 新pytorch再max(2)之后，已经没有了第2个维度。
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)    #sim_preds是预测出的字符串,类似“ XG78233838 ”


    for pred, name in zip(sim_preds, cpu_img_name):
      img_name_List.extend(str(name))
      img_pred_List.extend(str(pred))

  img_pred_List = np.array(img_pred_List)
  img_name_List = np.array(img_name_List)

  df = pd.DataFrame({'name':img_name_List, 'label':img_pred_List})
  column_order = ['name','label']
  df = df[column_order]
  predictionFile = '../../../../dataset_formal/classify_data/crnnData/result/crnn_result.csv'
  df.to_csv(predictionFile, index=False)

  print("\nover")

  # raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:666]   #raw_preds是长的那种....，类似 “XG--------7-8-2-3--3-8-3-8”










if __name__ == "__main__":
  test_lmdb_path = "../../../../dataset_formal/classify_data/crnnData/testDataLMDB"

  test_dataset = dataset.lmdbDataset(root=test_lmdb_path, type="test")

  nclass = len(alphabet) + 1
  nc = 1
  converter = utils.strLabelConverter(alphabet)

  crnn = crnn.CRNN(32, nc, nclass, 256)

  image = torch.FloatTensor(64, 3, 32, 32)
  text = torch.IntTensor(64 * 5)
  length = torch.IntTensor(64)

  if use_cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(1))
    image = image.cuda()

  print('loading pretrained model from %s' % saved_model_path)
  crnn.load_state_dict(torch.load(saved_model_path))
  print(crnn)

  # image = Variable(image)
  # text = Variable(text)
  # length = Variable(length)


  test_by_xzy(crnn,test_dataset)
