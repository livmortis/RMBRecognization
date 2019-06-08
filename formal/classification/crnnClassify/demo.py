import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)       #xzy [26, 1, 37]
# print("preds shape is: "+str(preds.shape))
# print("preds is: "+str(preds))

_, preds = preds.max(2)                    #xzy [26, 1]
# print("preds after max is: "+str(preds))
# print("preds after max shape is: "+str(preds.shape))

# print("preds shape is: "+str(preds.shape))
print("preds after transpose shape is: "+str(preds.transpose(1, 0).shape))    #xzy [1, 26]
# print("preds after transpose and contiguous shape is: "+str(preds.transpose(1, 0).contiguous().shape))
# print("preds after transpose and contiguous and view shape is: "+str(preds.transpose(1, 0).contiguous().view(-1).shape))
preds = preds.transpose(1, 0).contiguous().view(-1)       #xzy [26]
# print("preds after transpose data is: "+str(preds.data))


preds_size = Variable(torch.IntTensor([preds.size(0)]))   #xzy[ 26]
print("preds_size is: "+str(preds_size.data))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
