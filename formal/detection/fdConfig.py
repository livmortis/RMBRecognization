arti_img_path = "../../../dataset_formal/detect_data/arti_labeled_img_300/"
arti_label_path = "../../../dataset_formal/detect_data/arti_labeled_label_300/"
arti_txt_path = "../../../dataset_formal/detect_data/arti_labeled_txt_300/"
train_img_path = "../../../dataset_warm_up/train_data/"
model_saved = "../../../dataset_formal/detect_data/saved_model/"
output_reg_path = "../../../dataset_formal/detect_data/output_Reg/"
polyImg_reg_path = "../../../dataset_formal/detect_data/polyImg_Reg/"

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

lr_patient = 15
lr_shrink_factor = 0.5
LR = 0.0001

use_gpu = False
is_test = True
test_test_num = 10
test_train_num = 10
need_load_model = False

