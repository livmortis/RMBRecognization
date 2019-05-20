import wudata
import wuconfig
import torch.utils.data as Data
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":
  testDataset = wudata.datasetClass("test")
  testDataLoader = Data.DataLoader(testDataset, wuconfig.batch_size, shuffle=False)
  model_path = wuconfig.model_saved_path + "epoch_"+str(wuconfig.used_to_test_model_num)+"_saved_model.pkl"
  model = torch.load(model_path)

  preds = []
  image_names = []
  print("\nbegin to test")
  for index , (x, y) in tqdm(enumerate(testDataLoader, 0)):
    if wuconfig.USE_GPU:
      x = x.cuda()
      model = model.cuda()
    pred = model(x)
    preds.extend(pred)
    image_names.extend(y)

  print("\ntest have done")
  # 预测向量变为预测标签索引
  pred_rmb = []
  for one_pred in preds:
    if wuconfig.USE_GPU:
      one_pred_np = one_pred.detach().cpu().numpy()
    else:
      one_pred_np = one_pred.detach().numpy()
    one_index = np.argmax(one_pred_np)
    # 预测标签索引变为预测人民币面值
    one_rmb = wuconfig.map_dict_reverse[one_index]
    pred_rmb.append(one_rmb)
  pred_rmb = np.asarray(pred_rmb)
  image_names = np.asarray(image_names)

  # 预测值与图片名合并csv
  df = pd.DataFrame({'name':image_names,'label':pred_rmb})
  column_order = ['name','label']
  df = df[column_order]
  df.to_csv(wuconfig.pred_result_file)

  print("\nover")