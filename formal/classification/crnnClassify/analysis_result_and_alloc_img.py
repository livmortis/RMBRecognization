
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm



def wrong_length_distribute():
  # 分析108个长度错误的结果，所属面值.
  wrong_len_path = '../../../../dataset_formal/classify_data/crnnData/result/wrong_length_pred_6_19_score97'
  warm_test_result = '../../../../dataset_warm_up/pred_result/result-final-score100.csv'

  test_result_df = pd.read_csv(warm_test_result)
  wrong_list = os.listdir(wrong_len_path)
  dict = {"0.1":0 , "0.2":0, "0.5":0, "1.0":0, "2.0":0, "5.0":0, "10.0":0, "50.0":0, "100.0":0}
  for wrong in wrong_list:
    type = test_result_df.loc[test_result_df['name']== str(wrong) ]['label'].values[0]
    type = str(type)
    dict[type] = dict[type]+1
  print(dict)
  keys = tuple(dict.keys())
  values = tuple(dict.values())
  index = np.arange(len(values))
  plt.xticks(index, keys)
  plt.bar(index, values)
  plt.show()



def allocate_300_img_to_9_file():
  warm_train_result = '../../../../dataset_warm_up/train_face_value_label.csv'

  img300_path = '../../../../dataset_formal/detect_data/arti_labeled_img_300'

  img_0_1 = '../../../../dataset_formal/detect_data/CTPNData/0_1_img'
  img_0_2 = '../../../../dataset_formal/detect_data/CTPNData/0_2_img'
  img_0_5 = '../../../../dataset_formal/detect_data/CTPNData/0_5_img'
  img_1 = '../../../../dataset_formal/detect_data/CTPNData/1_img'
  img_2 = '../../../../dataset_formal/detect_data/CTPNData/2_img'
  img_5 = '../../../../dataset_formal/detect_data/CTPNData/5_img'
  img_10 = '../../../../dataset_formal/detect_data/CTPNData/10_img'
  img_50 = '../../../../dataset_formal/detect_data/CTPNData/50_img'
  img_100 = '../../../../dataset_formal/detect_data/CTPNData/100_img'

  train_result_df = pd.read_csv(warm_train_result)

  img300List = os.listdir(img300_path)
  for img in img300List:
    print(img)
    type = train_result_df.loc[train_result_df['name'] == str(img)][' label'].values[0]
    img =  '../../../../dataset_formal/detect_data/arti_labeled_img_300/'+str(img)
    print(type)
    if type == 0.1:
      shutil.copy(img, img_0_1)
    elif type == 0.2:
      shutil.copy(img, img_0_2)
    elif type == 0.5:
      shutil.copy(img, img_0_5)
    elif type == 1:
      shutil.copy(img, img_1)
    elif type == 2:
      shutil.copy(img, img_2)
    elif type == 5:
      shutil.copy(img, img_5)
    elif type == 10:
      shutil.copy(img, img_10)
    elif type == 50:
      shutil.copy(img, img_50)
    elif type == 100:
      shutil.copy(img, img_100)

  print("done")


def allocate_300_label_to_9_file():
  warm_train_result = '../../../../dataset_warm_up/train_face_value_label.csv'

  label300_path = '../../../../dataset_formal/detect_data/arti_labeled_txt_300'

  label_0_1 = '../../../../dataset_formal/detect_data/CTPNData/0_1_label'
  label_0_2 = '../../../../dataset_formal/detect_data/CTPNData/0_2_label'
  label_0_5 = '../../../../dataset_formal/detect_data/CTPNData/0_5_label'
  label_1 = '../../../../dataset_formal/detect_data/CTPNData/1_label'
  label_2 = '../../../../dataset_formal/detect_data/CTPNData/2_label'
  label_5 = '../../../../dataset_formal/detect_data/CTPNData/5_label'
  label_10 = '../../../../dataset_formal/detect_data/CTPNData/10_label'
  label_50 = '../../../../dataset_formal/detect_data/CTPNData/50_label'
  label_100 = '../../../../dataset_formal/detect_data/CTPNData/100_label'

  train_result_df = pd.read_csv(warm_train_result)

  label300List = os.listdir(label300_path)
  for label in label300List:
    label_belong_img = str(label).split(".")[0]+".jpg"
    type = train_result_df.loc[train_result_df['name'] == str(label_belong_img)][' label'].values[0]
    label =  '../../../../dataset_formal/detect_data/arti_labeled_txt_300/'+str(label)
    print(type)
    if type == 0.1:
      shutil.copy(label, label_0_1)
    elif type == 0.2:
      shutil.copy(label, label_0_2)
    elif type == 0.5:
      shutil.copy(label, label_0_5)
    elif type == 1:
      shutil.copy(label, label_1)
    elif type == 2:
      shutil.copy(label, label_2)
    elif type == 5:
      shutil.copy(label, label_5)
    elif type == 10:
      shutil.copy(label, label_10)
    elif type == 50:
      shutil.copy(label, label_50)
    elif type == 100:
      shutil.copy(label, label_100)

  print("done")




'''1500张图片'''



def allocate_1500_img_to_9_file():
  warm_train_result = '../../../../dataset_warm_up/train_face_value_label.csv'

  img1500_path = '../../../../dataset_formal/detect_data/arti_labeled_img_1500'

  img_0_1 = '../../../../dataset_formal/detect_data/CTPNData/0_1_img'
  img_0_2 = '../../../../dataset_formal/detect_data/CTPNData/0_2_img'
  img_0_5 = '../../../../dataset_formal/detect_data/CTPNData/0_5_img'
  img_1 = '../../../../dataset_formal/detect_data/CTPNData/1_img'
  img_2 = '../../../../dataset_formal/detect_data/CTPNData/2_img'
  img_5 = '../../../../dataset_formal/detect_data/CTPNData/5_img'
  img_10 = '../../../../dataset_formal/detect_data/CTPNData/10_img'
  img_50 = '../../../../dataset_formal/detect_data/CTPNData/50_img'
  img_100 = '../../../../dataset_formal/detect_data/CTPNData/100_img'

  train_result_df = pd.read_csv(warm_train_result)

  img1500List = os.listdir(img1500_path)
  for img in tqdm(img1500List):
    print(img)
    type = train_result_df.loc[train_result_df['name'] == str(img)][' label'].values[0]
    img =  '../../../../dataset_formal/detect_data/arti_labeled_img_1500/'+str(img)
    # print(type)
    if type == 0.1:
      shutil.copy(img, img_0_1)
    elif type == 0.2:
      shutil.copy(img, img_0_2)
    elif type == 0.5:
      shutil.copy(img, img_0_5)
    elif type == 1:
      shutil.copy(img, img_1)
    elif type == 2:
      shutil.copy(img, img_2)
    elif type == 5:
      shutil.copy(img, img_5)
    elif type == 10:
      shutil.copy(img, img_10)
    elif type == 50:
      shutil.copy(img, img_50)
    elif type == 100:
      shutil.copy(img, img_100)

  print("done")



def allocate_1500_label_to_9_file():
  warm_train_result = '../../../../dataset_warm_up/train_face_value_label.csv'

  label1500_path = '../../../../dataset_formal/detect_data/arti_labeled_txt_1500'

  label_0_1 = '../../../../dataset_formal/detect_data/CTPNData/0_1_label'
  label_0_2 = '../../../../dataset_formal/detect_data/CTPNData/0_2_label'
  label_0_5 = '../../../../dataset_formal/detect_data/CTPNData/0_5_label'
  label_1 = '../../../../dataset_formal/detect_data/CTPNData/1_label'
  label_2 = '../../../../dataset_formal/detect_data/CTPNData/2_label'
  label_5 = '../../../../dataset_formal/detect_data/CTPNData/5_label'
  label_10 = '../../../../dataset_formal/detect_data/CTPNData/10_label'
  label_50 = '../../../../dataset_formal/detect_data/CTPNData/50_label'
  label_100 = '../../../../dataset_formal/detect_data/CTPNData/100_label'

  train_result_df = pd.read_csv(warm_train_result)

  label1500List = os.listdir(label1500_path)
  for label in label1500List:
    label_belong_img = str(label).split(".")[0]+".jpg"
    type = train_result_df.loc[train_result_df['name'] == str(label_belong_img)][' label'].values[0]
    label =  '../../../../dataset_formal/detect_data/arti_labeled_txt_1500/'+str(label)
    # print(type)
    if type == 0.1:
      shutil.copy(label, label_0_1)
    elif type == 0.2:
      shutil.copy(label, label_0_2)
    elif type == 0.5:
      shutil.copy(label, label_0_5)
    elif type == 1:
      shutil.copy(label, label_1)
    elif type == 2:
      shutil.copy(label, label_2)
    elif type == 5:
      shutil.copy(label, label_5)
    elif type == 10:
      shutil.copy(label, label_10)
    elif type == 50:
      shutil.copy(label, label_50)
    elif type == 100:
      shutil.copy(label, label_100)

  print("done")




if __name__ == "__main__":
  # wrong_length_distribute()
  # allocate_300_label_to_9_file()
  # allocate_1500_img_to_9_file()
  allocate_1500_label_to_9_file()












