import dlib
import numpy as np
import cv2
import face_recognition
import sys
import dlib
import cv2
import os
import glob
import numpy as np
from pil import Image, ImageDraw, ImageFont

flag = 0
# 模型路徑

predictor_path = "model\\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "model\\dlib_face_recognition_resnet_model_v1.dat"
#測試圖片路徑
faces_folder_path = "test\\g.jpg"

# 讀入模型
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# 圖片辨識
img = cv2.imread('test/edward.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
faces = detector(gray, 1)

print(detector)

print("發現{0}個人臉".format(len(faces)))

for (index, face) in enumerate(faces):
    shape = shape_predictor(img, face)
    # print(predictor(img, face), "\n")
    for (index, point) in enumerate(shape.parts()):
        # cv2.rectangle(img, (face.left(), face.top(), face.right(), face.bottom()), (0, 255, 0), 1)
        cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), 1)
        flag = flag + 1
        # print(flag, ' : ', point.x, point.y)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)   # 計算人臉的128維的向量
        face_line = []
        for i in range(0, 128):
            face_line.append(face_descriptor[i])

print(face_line)
print(type(face_line))

        # for i in face_descriptor:
        #     print(face_descriptor(i))

cv2.namedWindow('face', cv2.WINDOW_FULLSCREEN)
cv2.imshow('face', img)
cv2.waitKey(0)
cv2.destroyWindow()

# 如果按下ESC键，就退出
# if cv2.waitKey(10) == 27:
#     cv2.destroyWindow('face')