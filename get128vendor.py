# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
import os
import glob
import numpy as np
from pil import Image, ImageDraw, ImageFont
import face_recognition

# 模型路徑

predictor_path = 'model\\shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'model\\dlib_face_recognition_resnet_model_v1.dat'
#測試圖片路徑
face_image_path = 'input\\Tzu-yu.jpg'

# 讀入模型
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

for img_path in glob.glob(os.path.join(face_image_path)):
    print('Processing file: {}'.format(img_path))
    # opencv 讀取圖片，並顯示
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # opencv的bgr格式圖片轉換成rgb格式
    b, g, r = cv2.split(img)
    imgrgb = cv2.merge([r, g, b])

    dets = detector(img, 1)   # 人臉標定
    print('Number of faces detected: {}'.format(len(dets)))

    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

        shape = shape_predictor(imgrgb, face)   # 提取68個特徵點
        for i, pt in enumerate(shape.parts()):
            #print('Part {}: {}'.format(i, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
            #print(type(pt))
        #print('Part 0: {}, Part 1: {} ...'.format(shape.part(0), shape.part(1)))
        cv2.namedWindow(img_path+str(index), cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path+str(index), img)

        face_descriptor = face_rec_model.compute_face_descriptor(imgrgb, shape)   # 計算人臉的128維的向量
        face_line = []
        for i in range(0, 128):
            face_line.append(face_descriptor[i])
        # print(face_line)

        # 新建一個和原圖一樣大小的全白圖片，用ImageDraw在上面勾出人臉輪廓，作為mask的模板
        mask = np.ones(imgrgb.shape, dtype=np.uint8) * 255
        mask = Image.fromarray(mask)
        q = ImageDraw.Draw(mask)
        q.shape(face_line)
        # q.line(face_line, width=5, fill=(255, 255, 255))
        # q.line(face_descriptor, width=5, fill=(255, 255, 255))
        # mask.show()
        mask.save('output\\mask.jpg')  # 將圖片寫出是為了交給OpenCV處理
        mask.show('output\\mask.jpg')
        # 生成掩膜
        # mask = cv2.imread('output\\mask.jpg')
        # h, w = mask.shape[:2]  # 读取图像的宽和高
        # mask_flood = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
        # mask = cv2.floodFill(mask, mask_flood, (75, 75), (0, 0, 0))  # 这里使用OpenCV的水漫填充，把轮廓外部涂成黑色，内部为白色



        # 用一个5*5的卷积核对掩膜进行闭运算，去掉噪声
        # erosion的迭代次数2次是我试出来的，感觉效果比较好
        # kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(mask, kernel, iterations=2)
        # dilation = cv2.dilate(erosion, kernel, iterations=1)
        # mask = dilation

k = cv2.waitKey(0)
cv2.destroyAllWindows()