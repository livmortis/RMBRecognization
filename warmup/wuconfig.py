map_dict_forward = {0.1: 0,
                     0.2: 1,
                     0.5: 2,
                     1: 3,
                     2: 4,
                     5: 5,
                     10: 6,
                     50: 7,
                     100: 8}

map_dict_reverse = { 0: 0.1,
                     1: 0.2,
                     2: 0.5,
                     3: 1,
                     4: 2,
                     5: 5,
                     6: 10,
                     7: 50,
                     8: 100}
value_num = len(map_dict_reverse)

train_data_path = '../../dataset_warm_up/train_data/'
test_data_path = '../../dataset_warm_up/public_test_data/'
csv_file = '../../dataset_warm_up/train_face_value_label.csv'
model_saved_path = '../../dataset_warm_up/saved_model/'
pred_result_file = '../../dataset_warm_up/pred_result/result.csv'
trainData_npy_saved_file =  '../../dataset_warm_up/saved_dataNpy_model/trainData.npy'
trainLabel_npy_saved_file =  '../../dataset_warm_up/saved_dataNpy_model/trainLabel.npy'
testData_npy_saved_file =  '../../dataset_warm_up/saved_dataNpy_model/testData.npy'
testName_npy_saved_file =  '../../dataset_warm_up/saved_dataNpy_model/testName.npy'
cm_saved_file =  '../../dataset_warm_up/saved_confusionMatrix/cm.npy'
image_size1 = 224
image_size2 = 324
batch_size = 64  #64最佳
# batch_size = 32
lr = 0.0001
weight_decay =  0.0001
lr_exponential_gamma = 0.9
epoch = 15


# 训练时需要修改：
TEST_WITH_LITTLE_DATA =True
TEST_NUM = 50
USE_GPU = False
EXIST_TRAIN_DATA_NPY = False    # 只在云主机上第一次训练读取数据时，设为False。之后.npy文件只要存在，都设为True。
EXIST_TEST_DATA_NPY = False    # 只在云主机上第一次测试读取数据时，设为False。之后.npy文件只要存在，都设为True。

newest_model_num = 0  # 训练用，手动更改为当前已有的最新模型编号！
used_to_test_model_num = 8 # 测试用，手动更改为需要用来测试的模型的编号！
