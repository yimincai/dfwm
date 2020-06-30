# -*- coding: utf-8 -*-

import dlib
import numpy
import cv2
from skimage import io

predictor_path = "model/shape_predictor_68_face_landmarks.dat"
faces_path = "input/Tzu-yu.jpg"

'''載入人臉檢測器、載入官方提供的模型構建特徵提取器'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
img = io.imread(faces_path)

win.clear_overlay()
win.set_image(img)

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

for k, d in enumerate(dets):
    shape = predictor(img, d)
    landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    print("face_landmark:")
    print(landmark)  # 列印關鍵點矩陣
    win.add_overlay(shape)  #繪製特徵點
    for idx, point in enumerate(landmark):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.3, color=(0, 255, 0))
        # cv2.circle(img, pos, 3, color=(0, 255, 0))
    win.set_image(img)

dlib.hit_enter_to_continue()
# Name of the output file
outputNameofImage = "output/image.jpg"
print("Saving output image to", outputNameofImage)
cv2.imwrite(outputNameofImage, img)

cv2.imshow("Face landmark result", img)

# Pause screen to wait key from user to see result
cv2.waitKey(0)
cv2.destroyAllWindows()