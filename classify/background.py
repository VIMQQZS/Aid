import cv2
img = cv2.imread('background.png',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
img1 = cv2.resize(img,(500,500))#调整大小
cv2.imwrite('bg.png',img1)#保存图片
print('转灰度成功！')