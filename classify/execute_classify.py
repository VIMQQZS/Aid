import cv2

watch_cascade = cv2.CascadeClassifier('/home/pi/opencv_createsamples/data/cascade.xml')#分类器路径

img = cv2.imread('test.jpg')#需要识别的照片，放到opencv_createsamples文件夹下

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

watches = watch_cascade.detectMultiScale(gray)

for (x,y,w,h) in watches:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)#建立方框，(0,255,0)表示绿色

    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]

cv2.imshow('识别窗口',img)
k = cv2.waitKey(0)
