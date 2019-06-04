
import os
import fdConfig
import cv2



def viewPolyInImg():
  outList = os.listdir(fdConfig.output_reg_path)


  rightLen = 20000 if not fdConfig.is_test else fdConfig.test_test_num
  if (len(outList) != rightLen):
    print("length error!： "+str(len(outList)))
    raise RuntimeError('length error!')
  for outname in outList:
    pureName = str(outname).split(".")[0]
    imgName = pureName+".jpg"
    txtName = pureName+".txt"
    img = cv2.imread(fdConfig.train_img_path+imgName)

    stream = open(fdConfig.output_reg_path+txtName)
    poly = stream.read()
    stream.close()
    polyList = poly.split(',')


    cv2.imshow(img)
    cv2.waitKey(0)





if __name__ == "__main__":
  viewPolyInImg()