
import  os
import lmdb


a = "../../../../dataset_formal/detect_data/polyImg_Reg-gpu_pad5455/"
b = os.listdir(a)
# print(b)
c = lmdb.open(a, max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

print(type(c))