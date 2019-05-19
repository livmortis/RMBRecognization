import torch.utils.data as Data
import torch as torch
import pandas as pd
import wuconfig
import numpy as np
from tqdm import tqdm
# import cv2
import PIL.Image as Image
import os


#共有9种面值，分别编号为 0.1，0.2，0.5，1，2，5，10，50，100
def label_to_onehot(label_np):

  if wuconfig.TEST_without_GPU:
    label_np = label_np[0:wuconfig.TEST_NUM]

  # crossEntropyLoss不需要one-hot.....
  # labels_np_hot = []
  # for label in label_np:
  #   label_mask = np.zeros(9)
  #   index = wuconfig.map_dict_forward[label]
  #   label_mask[index] = 1
  #   labels_np_hot.append(label_mask)

  labels_np_index = []
  for label in label_np:
    index = wuconfig.map_dict_forward[label]
    index = np.asarray(index)
    labels_np_index.append(index)


  # return labels_np_hot
  return labels_np_index



def read_csv ():
  csv_file = wuconfig.csv_file
  csv_df = pd.read_csv(csv_file)
  print(csv_df.head())

  # 收集数据名字
  pic_name_df = csv_df['name']
  pic_name_np = np.asarray(pic_name_df)


  # 收集数据标签，并转onehot
  label_df = csv_df[' label']
  label_np = np.asarray(label_df)
  label_hot = label_to_onehot(label_np)


  return pic_name_np, label_hot

def load_test():
  test_data_path = wuconfig.test_data_path
  test_name_list = os.listdir(test_data_path)
  # if wuconfig.TEST_without_GPU:
  #   test_name_list = test_name_list[0:wuconfig.TEST_NUM]
  test_images = []
  test_name_np_list = []
  # test_name_np_list = np.zeros( [len(test_name_list),1 ], dtype=np.str )
  print("begin to load test data")
  i = 0
  for one_test_name in tqdm(test_name_list):
    try:
      one_test_image = Image.open(test_data_path + str(one_test_name))
    except:
      # PLYNBX7G.jpg为空
      print("\nthis picture is broken: "+ str(one_test_name))
      one_test_image = Image.open(test_data_path + "0A2MIKHN.jpg")  #随便挑选第一张替换

    if one_test_image.mode != "RGB":
      one_test_image.convert("RGB")
    one_test_image = one_test_image.resize([wuconfig.image_size,wuconfig.image_size])
    one_test_np = np.asarray(one_test_image)
    one_test_np = one_test_np.transpose([2,0,1])
    # print("11111"+str(type(one_test_np)))
    test_images.append(one_test_np)
    # print("333333"+str(type(one_test_name)))

    one_test_name_np = np.asarray(one_test_name)
    # print("22222"+str(type(one_test_name_np)))
    test_name_np_list.append(one_test_name_np)
    # test_name_np_list[i, :] = one_test_name_np
    i+=1
  test_images = np.asarray(test_images)
  test_name_np_list = np.asarray(test_name_np_list)
  return test_images, test_name_np_list




def load_train ():
  train_data_path = wuconfig.train_data_path
  pic_names ,label_index= read_csv()

  if wuconfig.TEST_without_GPU:
    pic_names = pic_names[0:wuconfig.TEST_NUM]
  train_images = []
  print("begin to load image")
  for i in tqdm(range(pic_names.size)):
    # cv2
    # train_image_cv = cv2.imread(train_data_path + str(pic_names[i]))
    # if (train_image_cv.shape[2] != 3):
    #   print('this image is not three channel: ' + str(pic_names[i]))
    # train_image_cv = cv2.resize(train_image_cv, (wuconfig.image_size, wuconfig.image_size), interpolation=cv2.INTER_AREA)  #cv2

    # PIL
    train_image_cv = Image.open(train_data_path + str(pic_names[i]))
    if train_image_cv.mode != "RGB":
      train_image_cv.covert("RGB")
    train_image_cv = train_image_cv.resize((wuconfig.image_size, wuconfig.image_size))

    train_np = np.asarray(train_image_cv)
    train_np = train_np.transpose([2,0,1])  #一般读取图片为HWC，pytorch需要该成CHW。
    train_images.append(train_np)
  # print(train_images[0].shape)
  train_images_np = np.asarray(train_images)
  return train_images_np, label_index

class datasetClass(Data.Dataset):
    def __init__(self, type):
      self.t = type
      if type == "test":
        test_images_np, test_name_list_np = load_test()
        self.l = len(test_images_np)
        self.x = test_images_np
        self.y = test_name_list_np

      else:
        train_valid_images_np, label_index = load_train()
        total_num = len(label_index)
        train_images_np = train_valid_images_np[: int(total_num * 0.7)]
        train_label_index = label_index[: int(total_num * 0.7)]
        valid_images_np = train_valid_images_np[int(total_num * 0.7):]
        valid_label_index = label_index[int(total_num * 0.7):]

        if type=="train":
          self.l = len(train_images_np)
          self.x = train_images_np
          self.y = train_label_index

        elif type =="valid":

          self.l = len(valid_images_np)
          self.x = valid_images_np
          self.y = valid_label_index



    def __len__(self):
        return self.l

    def __getitem__(self, index):
      x = torch.from_numpy(self.x[index])
      x = x.type(torch.FloatTensor)
      if self.t == "test":
        y = self.y[index]
      else:
        y = torch.from_numpy(self.y[index])
        y = y.type(torch.LongTensor)
      return x , y





if __name__ == "__main__":
  # train_images = load_train()
  # pic_names ,label_np= read_csv()
  # label_to_onehot(label_np)
  # load_test()
  print('well down')
