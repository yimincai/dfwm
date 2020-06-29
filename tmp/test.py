import face_recognition
import numpy as np
import cv2
from pil import Image, ImageDraw, ImageFont

image = face_recognition.load_image_file("input\\Tzu-yu.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

face_line = list()#用于记录脸部轮廓
face_point = list()#用于记录图片需要截取的范围
face = list()#用于记录脸

#获取脸部最高点
def the_top(x_list):
    return sorted(x_list, key=lambda x: x[1])

#获取脸部最低点
def the_bottom(x_list):
    return sorted(x_list, key=lambda x: x[1], reverse=True)

#获取脸部最左点
def the_left(y_list):
    return sorted(y_list, key=lambda y: y[0])

#获取脸部最右点
def the_right(y_list):
    return sorted(y_list, key=lambda y: y[0], reverse=True)

#计算正方形左上角和右下角的点
#表情位置用正方形把它截下来
def square(face_point):
    centre, top, bottom, left, right = face_point['centre'], face_point['top'], face_point['bottom'], face_point['left'], face_point['right']
    #用眉毛以及下巴作为选框参照
    # side = ((centre[1] - top[1]) if ((centre[1] - top[1]) > (bottom[1] - centre[1])) else (bottom[1] - centre[1]))
    #用左右轮廓作为选框参照
    side = ((centre[0] - left[0]) if ((centre[0] - left[0]) < (right[0] - centre[0])) else (right[0] - centre[0]))
    return (centre[0] - side, centre[1] - side), (centre[0] + side, centre[1] + side)

#锐化函数
def custom_blur_demo(image):
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(image, -1, kernel=kernel1)
    return dst

for face_landmarks in face_landmarks_list:

    #找出每张脸的轮廓用于生成掩膜
    chin = face_landmarks['chin']
    left_eyebrow = face_landmarks['left_eyebrow']
    right_eyebrow = face_landmarks['right_eyebrow']
    nose_bridge = face_landmarks['nose_bridge']
    chin.reverse()
    list_all = left_eyebrow + right_eyebrow + chin + [left_eyebrow[0]]
    face_line.append(list_all)

    #找到每张脸最左端点，最右端点，最高点，最低点以及中心，用于最大限度截取表情
    face_five_point = {'centre': nose_bridge[len(nose_bridge)-1],
                       'top': the_top(left_eyebrow + right_eyebrow)[0],
                       'bottom': the_bottom(chin)[0],
                       'left': the_left(chin)[0],
                       'right': the_right(chin)[0]
                       }
    start, end = square(face_five_point)
    face_point.append((start[0], start[1], end[0], end[1]))

   #让我们在图像中描绘出每个人脸特征！
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    chin = face_landmarks['chin']
    left_eyebrow = face_landmarks['left_eyebrow']
    right_eyebrow = face_landmarks['right_eyebrow']
    chin.reverse()
    list_all = left_eyebrow + right_eyebrow + chin + [left_eyebrow[0]]
    face_line.append(list_all)
    d.line(list_all, width=5)
    pil_image.save("Facial_contour.jpg")
    pil_image.show("Facial_contour.jpg")

#新建一个和原图一样大小的全黑图片，用ImageDraw在上面勾出人脸轮廓，作为掩膜的模板
mask = np.zeros(image.shape, dtype=np.uint8)
mask = Image.fromarray(mask)
q = ImageDraw.Draw(mask)
for i in face_line:
    q.line(i, width=5, fill=(0, 0, 0))
mask.save(r"output/mask.jpg")#将图片写出是为了交给OpenCV处理

# 生成掩膜
mask = cv2.imread('output/mask.jpg')
h, w = mask.shape[:2]  # 读取图像的宽和高
mask_flood = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
cv2.floodFill(mask, mask_flood, (75, 75), (0, 0, 0))  # 这里使用OpenCV的水漫填充，把轮廓外部涂成黑色，内部为白色

# 用一个5*5的卷积核对掩膜进行闭运算，去掉噪声
# erosion的迭代次数2次是我试出来的，感觉效果比较好
# kernel = np.ones((5, 5), np.uint8)
# erosion = cv2.erode(mask, kernel, iterations=2)
# dilation = cv2.dilate(erosion, kernel, iterations=1)
# mask = dilation

#重新读入原图，框出RIO，交给OpenCV处理
# image = cv2.imread("input/Tzu-yu.jpg")
# image[mask == 0] = 0

#将处理过的图片变为灰度图
# GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#二值化处理
# ret, image = cv2.threshold(GrayImage, 80, 255, cv2.THRESH_BINARY)

#再来一次水漫填充，这次把轮廓之外的地方变成白色
h, w = image.shape[:2]
mask_flood = np.zeros([h + 2, w + 2], np.uint8)
image = cv2.floodFill(image, mask_flood, (1, 1), (255, 255, 255))
cv2.imwrite("output/last_out.jpg", image)

#用一个2*2的卷积核对RIO进行多次腐蚀膨胀处理，去掉噪声，这里进行多少次膨胀腐蚀要结合图片效果来进行
# kernel = np.ones((2,2),np.uint8)
# dilation = cv2.dilate(image,kernel,iterations=2)
# erosion = cv2.erode(dilation,kernel,iterations=1)
# dilation = cv2.dilate(image,kernel,iterations=1)
# image = cv2.erode(dilation,kernel,iterations=1)

#这是表情需要截取的部分，不同图这个地方的参数不同
# image = image[130:410, 170:505]
# cv2.imwrite("output/last.png",image)#这里输出图片是为了给Image做处理

# box = (160, 155, 465, 405)#背景图要被替换的部分
# base_img = Image.open('input/background.png')
# image = Image.open('output/last.png')
# image = image.resize((box[2] - box[0], box[3] - box[1]))#缩放表情，贴入的表情必须和背景被替换的地方大小相同
# base_img.paste(image, box)
# base_img.show()
# base_img.save('output/out.png')#这里输出图片是为了给cv2做处理

#这是为了让图片看起来更自然，而且本身表情就是黑白的，所以对图片再进行一次二值化处理
# image = cv2.imread(r'output/out.png')
# GrayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# ret,image = cv2.threshold(GrayImage,85,255,cv2.THRESH_BINARY)
#
#这里继续膨胀腐蚀，让图片更好看一点
# erosion = cv2.erode(image,kernel,iterations=1)
# dilation = cv2.dilate(erosion,kernel,iterations=2)
# erosion = cv2.erode(dilation,kernel,iterations=1)
# image = erosion
# cv2.imwrite("output/last_out.jpg", image)
cv2.waitKey(0)