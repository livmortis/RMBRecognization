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


train_data_path = '../../dataset_warm_up/train_data/'
test_data_path = '../../dataset_warm_up/public_test_data/'
csv_file = '../../dataset_warm_up/train_face_value_label.csv'
model_saved_path = '../../dataset_warm_up/saved_model/'
pred_result_file = '../../dataset_warm_up/pred_result/result.csv'
image_size = 224
# batch_size = 128
batch_size = 32
lr = 0.01
weight_decay =  0.0001
lr_exponential_gamma = 0.9
epoch = 15

TEST_without_GPU =True
TEST_NUM = 50



newest_model_num = 0  # 手动更改为当前已有的最新模型编号！
used_to_test_model_num = 0 #手动更改为需要用来测试的模型的编号！
