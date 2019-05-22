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

  if wuconfig.TEST_WITH_LITTLE_DATA:
    label_np = label_np[0:wuconfig.TEST_NUM]

  # 更新：弃用。  one-hot必须在dataloader读出来以后再改回index，那是数据已经是Tensor，再该还必须改回numpy——转index——改回Tensor，还涉及cuda，太不合理
  # 这是one-hot
  # crossEntropyLoss不需要one-hot.....
  # 但是，由于下面两者的原因，仍然得用one-hot过渡。
  # labels_np_hot = []
  # for label in label_np:
  #   label_mask = np.zeros(9)
  #   index = wuconfig.map_dict_forward[label]
  #   label_mask[index] = 1
  #   labels_np_hot.append(label_mask)


  # 更新：最终结果——采用。  label可以为list，但list的元素必须是numpy。 labels = list[np, np, np]是可以的，
  #       但labels转换一次numpy，就会变成labels = np([int64, int64, int64])，这样就会出错__getItem__中转换Tensor时出错。
  #       而存储npy文件意味着一定会转换一次numpy。  所以补救措施为，load出npy文件后，再遍历np，把每个元素变为np，即labels = np([np, np, np])，即可。
  # 这样会使labels_np_index的元素index为numpy.int64,而不是np.ndarray，转换Tensor时报错。弃用
  labels_np_index = []
  for label in label_np:
    index = wuconfig.map_dict_forward[label]
    index = np.asarray(index)
    # index = np.ndarray(index) #不行！！
    labels_np_index.append(index)


  # 更新：最终结果——弃用。  这样会使得label为二维numpy形式，在交叉熵损失函数中，label不能有二维。
  # 该成numpy.zeros扩展的方式死，仍然会使labels_np_index的元素index为numpy.int64,而不是np.ndarray，转换Tensor时报错。弃用——用回one-hot
  # labels_np_index = np.zeros([len(label_np), 1])
  # k = 0
  # for label in label_np:
  #   index = wuconfig.map_dict_forward[label]
  #   index = np.asarray(index)
  #   labels_np_index[k,:] = index
  #   k += 1


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
  print("6666666666"+str(test_name_list[0:10]))
  if wuconfig.TEST_WITH_LITTLE_DATA:
    test_name_list = test_name_list[0:wuconfig.TEST_NUM]
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
  print("777777"+str(test_name_np_list[0:10]))

  np.save(wuconfig.testData_npy_saved_file, test_images) # 存储数据以便下次使用
  np.save(wuconfig.testName_npy_saved_file, test_name_np_list) # 存储数据以便下次使用
  print ("\ntest npy has saved")

  return test_images, test_name_np_list




def load_train ():
  train_data_path = wuconfig.train_data_path
  pic_names ,label_index= read_csv()

  if wuconfig.TEST_WITH_LITTLE_DATA:
    pic_names = pic_names[0:wuconfig.TEST_NUM]
  train_images = []
  print("begin to load image")
  for i in tqdm(range(pic_names.size)):
    try:
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
    except:
      print("\n image broken when load trainDataset: "+ str(pic_names[i]))
      replace_image = Image.open(train_data_path + str(pic_names[0]))
      replace_image = replace_image.resize((wuconfig.image_size, wuconfig.image_size))
      replace_image = np.asarray(replace_image)
      replace_image = replace_image.transpose([2,0,1])
      train_images.append(replace_image)



  # print(train_images[0].shape)
  train_images_np = np.asarray(train_images)
  # label_index_np = np.asarray(label_index)

  np.save(wuconfig.trainData_npy_saved_file, train_images_np)  # 存储数据以便下次使用
  np.save(wuconfig.trainLabel_npy_saved_file, label_index)  # 存储数据以便下次使用
  print ("\ntrain npy has saved")
  return train_images_np, label_index

class datasetClass(Data.Dataset):
    def __init__(self, type):
      self.t = type
      if type == "test":
        if wuconfig.EXIST_TEST_DATA_NPY:
          test_images_np = np.load(wuconfig.testData_npy_saved_file)
          test_name_list_np = np.load(wuconfig.testName_npy_saved_file)
          print("\nsucess load test npy")

        else:
          test_images_np, test_name_list_np = load_test()
        self.l = len(test_images_np)
        self.x = test_images_np
        self.y = test_name_list_np

      else:
        if wuconfig.EXIST_TRAIN_DATA_NPY:   # 只要不是第一次在主机上训练，都可以跳过load_train()方法。
          train_valid_images_np = np.load(wuconfig.trainData_npy_saved_file)
          label_index_list = np.load(wuconfig.trainLabel_npy_saved_file)

          # 严重bug解决方法。
          label_index = []
          for i in label_index_list:
            i_np = np.asarray(i)
            label_index.append(i_np)

          print("\nsucess load train npy")
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
      # 关于x
      x = torch.from_numpy(self.x[index])
      x = x.type(torch.FloatTensor)

      # 关于y
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
  load_test()
  print('well down')
