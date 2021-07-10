import cv2
for i in range(1,9):#批量处理照片
	img = cv2.imread('pos/pos000'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
	img1 = cv2.resize(img,(500,500))#调整大小
	cv2.imwrite('pos/'+str(i)+'.png',img1)#保存图片
print('Finish part1')
for i in range(10,99):#批量处理照片
	img = cv2.imread('pos/pos00'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
	img1 = cv2.resize(img,(500,500))#调整大小
	cv2.imwrite('pos/'+str(i)+'.png',img1)#保存图片
print('Finish part2')
for i in range(100,999):#批量处理照片
	img = cv2.imread('pos/pos0'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
	img1 = cv2.resize(img,(500,500))#调整大小
	cv2.imwrite('pos/'+str(i)+'.png',img1)#保存图片
print('Finish part3')
for i in range(1000,6129):#批量处理照片
	img = cv2.imread('pos/pos'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
	img1 = cv2.resize(img,(500,500))#调整大小
	cv2.imwrite('pos/'+str(i)+'.png',img1)#保存图片
print('Finish part4')
print('批量转灰度成功！')