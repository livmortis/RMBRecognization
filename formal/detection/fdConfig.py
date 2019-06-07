arti_img_path = "../../../dataset_formal/detect_data/arti_labeled_img_300/"
arti_label_path = "../../../dataset_formal/detect_data/arti_labeled_label_300/"
arti_txt_path = "../../../dataset_formal/detect_data/arti_labeled_txt_300/"
train_img_path = "../../../dataset_warm_up/train_data/"
model_saved = "../../../dataset_formal/detect_data/saved_model/"
output_reg_path = "../../../dataset_formal/detect_data/output_Reg/"     # poly的txt坐标文件，训练集和测试集共用，测试测试集时手动清空。
polyImg_reg_path = "../../../dataset_formal/detect_data/polyImg_Reg/"   # poly裁剪后的jpg图片文件，训练集和测试集共用，测试测试集时手动清空。
test_img_path = "../../../dataset_warm_up/public_test_data/"


EPOCH = 50
IMG_SIZE_WIDTH = 224
IMG_SIZE_HEIGHT = 224
IMG_SIZE_EAST = 416
BATCH_SIZE = 16   #EAST时16以上会OOM
WHICH_MODEL = 'R'   #'R'是回归模型，'E'是EAST模型。
WEIGHT_DECAY =  0.0001
lr_exponential_gamma = 0.9
train_cal_iou_num = 15
LOG_FOR_NET_CONSTRUCTION = False
LOG_FOR_EAST_DATA = False
LOG_FOR_EAST_MODEL = False
LOG_FOR_EAST_LOSS = False
LOG_FOR_EAST_TEST = True

east_detect_scoremap_thresh = 0.9

lr_patient = 15
lr_shrink_factor = 0.5
LR = 0.0001

use_gpu = False
is_test = False
test_test_num = 100
test_train_num = 10
need_load_model = False

# MODEL_NAME = "detect_reg_model2.pkl"
# MODEL_NAME = "detect_east_model_cpu_loss0-0113.pkl"
# MODEL_NAME = "detect_reg_model_cpu_loss15_epo8.pkl"
MODEL_NAME = "detect_reg_model6.pkl"    # gpu云上目前最好模型 2019.6.7

# detect_poly_of_train_or_test = "detect_train"    # 检测算法，检测训练集的poly还是测试集的poly
detect_poly_of_train_or_test = "detect_test"