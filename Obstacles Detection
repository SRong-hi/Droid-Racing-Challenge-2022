import cv2
import numpy as np
#obstacle size max 400x400x500mm

camera = cv2.VideoCapture(0)  #打开默认摄像头采集图像
if (camera.isOpened()):
  print('Camera Open')
else:
  print('Camera Close')

size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))

# width = 640  #定义摄像头获取图像宽度
# height = 480   #定义摄像头获取图像长度
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  #设置宽度
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  #设置长度


es = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 4))  #返回指定形状和尺寸的结构元素。
kernel = np.ones((5, 5), np.uint8)
background = None
 
while True:
    grabbed, frame_lwpCV = camera.read()
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)  #(21, 21)高斯核大小。 ksize.width 并且 ksize.height 可以有所不同，但它们都必须是正数和奇数
    
    if background is None:
        background = gray_lwpCV
        continue  
    diff = cv2.absdiff(background, gray_lwpCV)  #获取差分图 就是将两幅图像作差
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 对图像运用二值化处理
    diff = cv2.dilate(diff, es, iterations=2)  #cv2.dilate( img, kernel, iterations ) 迭代次数越多越膨胀

    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #cv2.retr_external表示只检测外轮廓
    #cv2.chain_approx_simple压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息。
    #image：是原图像;contours：图像的轮廓，以列表的形式表示，每个元素都是图像中的一个轮廓;hier：相应轮廓之间的关系。
    for c in contours:
        if cv2.contourArea(c) < 4000: 
            continue
        (x, y, w, h) = cv2.boundingRect(c)   #img是一个二值图，分别是x，y，w，h；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
        cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2) #画出矩行
    
    cv2.imshow('contours', frame_lwpCV)
    cv2.imshow('dis', diff)
 
    key = cv2.waitKey(1) & 0xFF  #cv2.waitKey(1) 1为参数，单位毫秒，表示间隔时间
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
 
