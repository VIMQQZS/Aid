import cv2
for  i in range(1,45):#批量处理照片
	img = cv2.imread('neg/'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
	img1 = cv2.resize(img,(500,500))#调整大小
	cv2.imwrite('neg/'+str(i)+'.jpg',img1)#保存图片
print('批量转灰度成功！')