
import os
import fdConfig
import cv2
from tqdm import tqdm

# type = "view"
type = "crop"


def viewOrCrop(type):
  outList = os.listdir(fdConfig.output_reg_path)


#  rightLen = 39620 if not fdConfig.is_test else fdConfig.test_test_num
#  if (len(outList) != rightLen):
#    print("length error!： "+str(len(outList)))
#    raise RuntimeError('length error!')
  for outname in tqdm(outList):
    pureName = str(outname).split(".")[0]
    imgName = pureName+".jpg"
    txtName = pureName+".txt"
    img = cv2.imread(fdConfig.train_img_path+imgName)

    stream = open(fdConfig.output_reg_path+txtName)
    poly = stream.read()
    stream.close()
    polyList = poly.split(',')
    min_x = int(float(polyList[0]))
    min_y = int(float(polyList[1]))
    max_x = int(float(polyList[2]))
    max_y = int(float(polyList[3]))

    if type == "view":
      # poly坐标是图片固定为224*448时的坐标，预览时将图片resize为224*448

      height_resized = fdConfig.IMG_SIZE_HEIGHT  # 224
      width_resized = int(height_resized * 2)  # 448
      img = cv2.resize(img, (width_resized, height_resized))

      cv2.rectangle(img, (min_x,min_y),(max_x,max_y),color=(255,255,255),thickness=1)
      cv2.imshow("a",img)
      cv2.waitKey(0)
    else:
      # poly坐标是图片固定为224*448时的坐标，裁剪时将poly缩放为真实图片大小的poly坐标

      height_acord_poly = fdConfig.IMG_SIZE_HEIGHT   # 224
      width_acord_poly = height_acord_poly * 2   # 448
      height = img.shape[0]
      width = img.shape[1]
      # print(height)
      # print(width)
      ratio_h = height/height_acord_poly
      ratio_w = width/width_acord_poly

      min_x *= ratio_w
      max_x *= ratio_w
      min_y *= ratio_h
      max_y *= ratio_h


      polyImg = img[ int(min_y-4):int(max_y+5),int(min_x-5):int(max_x+5)]
      # cv2.imshow("b", polyImg)
      # cv2.waitKey(0)
      cv2.imwrite(fdConfig.polyImg_reg_path + "poly_" + pureName + ".jpg", polyImg)




if __name__ == "__main__":
  viewOrCrop(type)
