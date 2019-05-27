arti_img_path = "../../../dataset_formal/detect_data/arti_labeled_img_300/"
arti_label_path = "../../../dataset_formal/detect_data/arti_labeled_label_300/"
arti_txt_path = "../../../dataset_formal/detect_data/arti_labeled_txt_300/"
train_img_path = "../../../dataset_warm_up/train_data/"
model_saved = "../../../dataset_formal/detect_data/saved_model/"

EPOCH = 10
IMG_SIZE_WIDTH = 224
IMG_SIZE_HEIGHT = 224
BATCH_SIZE = 32
WHICH_MODEL = 'R'   #'R'是回归模型，'E'是EAST模型。
LR = 0.001
WEIGHT_DECAY =  0.0001
lr_exponential_gamma = 0.9

use_gpu = False
is_test = True
need_load_model = False

