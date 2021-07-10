'''
在opencv_workspace文件夹下创建文件夹pos，将正样本放到pos下
将正样本图片转为灰度图，并剪成合适的尺寸，方便后续处理。创建1.py文件，放进下面代码并运行。
'''
import cv2
for  i in range(1,6):#批量处理照片
	img = cv2.imread('pos/'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)#读入照片，并转灰度
	img1 = cv2.resize(img,(50,50))#调整大小
	cv2.imwrite('pos/'+str(i)+'.jpg',img1)#保存图片
print('批量转灰度成功！')
'''
将负样本文件夹neg放到opencv_workspace文件夹下即可。
生成正样本描述文件，建立2.py文件，把下面代码拷贝进去
'''

import os

#此处修改pos或neg即可生成正负样本的描述文件，pos是生成正样本描述文件info.txt
def create_pos_n_neg():
    for file_type in ['pos']:
        for img in os.listdir(file_type):
            if (file_type == 'neg'):
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)
            elif (file_type == 'pos'):
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.txt', 'a') as f:
                    f.write(line)

if __name__ == '__main__':
    create_pos_n_neg()
    print('正样本描述文件info.txt已生成')

'''生成负样本描述文件，建立3.py文件，把下面代码拷贝进去'''
import os
def create_pos_n_neg():
    for file_type in ['neg']: #此处修改pos或neg即可生成正负样本的描述文件，neg是生成正样本描述文件bg.txt
        for img in os.listdir(file_type):
            if (file_type == 'neg'):
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)
            elif (file_type == 'pos'):
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.txt', 'a') as f:
                    f.write(line)

if __name__ == '__main__':
    create_pos_n_neg()
    print('负样本描述文件bg.txt已生成')

'''
三、训练分类器
生成positives.vec文件。
在当前目录打开控制台程序，输入
opencv_createsamples -info info.txt -num 50 -w 50 -h 50 -vec positives.vec
（其中，-info字段填写正样本描述文件；-num制定正样本的数目；-w和-h分别指定正样本的宽和高（-w和-h越大，训练耗时越大）；-vec用于保存制作的正样本。）
mkdir data #用于存储Cascade分类器数据

训练分类器
opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 4 -numNeg 10 -numStages 16 -w 50 -h 50


字段说明如下：
-data data：训练后data目录下会存储训练过程中生成的文件
-vec positives.vec：Pos.vec是通过opencv_createsamples生成的vec文件
-bg bg.txt：bg.txt是负样本文件的数据
-numPos ：正样本的数目，这个数值一定要比准备正样本时的数目少，不然会报can not get new positive sample.
-numNeg ：负样本数目，数值可以比负样本大
-numStages ：训练分类器的级数
-w 50：必须与opencv_createsample中使用的-w值一致
-h 50：必须与opencv_createsample中使用的-h值一致
注：-w和-h的大小对训练时间的影响非常大


进入data文件夹下，就可以看到生成训练出来的xml文件cascade.xml。

四、使用生成的xml文件进行识别
将需要识别的照片（命名为test.jpg）放到在opencv_createsamples文件夹下，并在opencv_createsamples文件夹下创建4.py文件，拷贝以下代码进去
'''

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

