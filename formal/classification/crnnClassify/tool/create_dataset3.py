import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import argparse
import pandas as pd
from tqdm import tqdm


def init_args():
    args = argparse.ArgumentParser()
    args.add_argument('-i',
                      '--image_dir',
                      type=str,
                      help='The directory of the dataset , which contains the images',
                      default='../../../../../dataset_formal/detect_data/polyImg_Reg-gpu_pad5455/')
    args.add_argument('-l',
                      '--label_file',
                      type=str,
                      help='The file which contains the paths and the labels of the data set',
                      default='../../../../../dataset_formal/classify_data/train_id_label.csv')
    args.add_argument('-s',
                      '--save_dir',
                      type=str
                      , help='The generated mdb file save dir',
                      # default='../../../../../dataset_formal/classify_data/crnnData/trainDataLMDB')
                      default='../../../../../dataset_formal/classify_data/crnnData/valDataLMDB')
    args.add_argument('-m',
                      '--map_size',
                      help='map size of lmdb',
                      type=int,
                      default=4000000000)

    return args.parse_args()



def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      if type(k) == str:
        k = k.encode()
      if type(v) == str:
        v = v.encode()
      txn.put(k, v)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  assert (len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=1099511627776)
  cache = {}
  cnt = 1
  for i in range(nSamples):
    imagePath = imagePathList[i]
    label = labelList[i]
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue

    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt - 1
  cache['num-samples'] = str(nSamples)
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    args = init_args()
    df = pd.read_csv(args.label_file)           #xzy  csv文件读取方法
    length = len(df)
    print(length)

    imgDir = args.image_dir
    imgPathList = []
    labelList = []

    i = 0              #xzy  csv文件读取方法
    # for i in tqdm(range(length)):      #制作训练集
    # for i in tqdm(range(2000)):       #制作验证集
    for i in tqdm(range(10)):       #实验
      imgPath = os.path.join(imgDir,df['name'][i])
      imgPathList.append(imgPath)
      word = df[' label'][i].strip()    #新增strip()—— 每个标签前都有空格
      labelList.append(word)
      print(imgPath)
      print(word)
    a = cv2.imread(imgPathList[0])
    cv2.imshow("a",a)
    cv2.waitKey(0)
    print(len(imgPathList))
    print(len(labelList))


    # createDataset(args.save_dir, imgPathList, labelList)