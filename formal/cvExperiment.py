import cv2

a = cv2.imread("../../dataset_warm_up/samplermb.jpg")
print(a.shape)
# a = cv2.resize(a,(247,120))
# a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
a = cv2.Canny(a,0,255)
# cv2.threshold(a, 0.5, 0, 255)
# cv2.dilate(a,(3,3),a )
# a = cv2.morphologyEx(a, cv2.MORPH_OPEN, (3,3))
cv2.imshow("a",a)
cv2.waitKey(0)