# -*- coding: utf-8 -*-

import dlib
import numpy
import cv2
import imutils
import face_recognition
import os
import shutil

import numpy as np
from pil import Image, ImageDraw
from skimage import io


def get_point(faces_path):
    predictor_path = "model/shape_predictor_68_face_landmarks.dat"
    # faces_path = "input/Tzu-yu.jpg"

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
        win.add_overlay(shape)  # 繪製特徵點
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.3, color=(0, 255, 0))
            # cv2.circle(img, pos, 3, color=(0, 255, 0))
        win.set_image(img)

    dlib.hit_enter_to_continue()
    # Name of the output file
    # outputNameOfImage = "output/image.jpg"
    # output_path = 'output/image.jpg'
    # print("Saving output image to", output_path)
    # cv2.imwrite(output_path, img)

    cv2.imshow("Face landmark result", img)

    # Pause screen to wait key from user to see result
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_face_rec(image_path):
    # 讀取照片圖檔
    img = cv2.imread(image_path)

    # 縮小圖片
    img = imutils.resize(img, width=1280)

    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()

    # 偵測人臉
    face_rects = detector(img, 0)

    # 取出所有偵測的結果
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        # 以方框標示偵測的人臉
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

    # 顯示結果
    cv2.imshow("Face Detection", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_mask(load_image_path):
    mask_path = 'output/mask.jpg'
    image_name = load_image_path.partition('.')[0].partition('/')[2]
    image_name_with_fe = load_image_path.partition('/')[2]
    output_dir = 'output/' + image_name + '/'
    os.makedirs(output_dir, exist_ok=True)
    image = face_recognition.load_image_file(load_image_path)
    face_locations = face_recognition.face_locations(image, model='cnn')
    top, right, bottom, left = face_locations[0]
    print(
        'A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}'.format(top, left, bottom, right))
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    # pil_image.show()
    pil_image.save(output_dir + '_face_detect_' + image_name_with_fe)

    face_landmarks_lists = face_recognition.face_landmarks(image)
    for face_landmarks in face_landmarks_lists:
        # white background
        mask = np.zeros(image.shape, np.uint8) + 255
        # black background
        # mask = np.zeros(image.shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        ImageDraw.Draw(mask)
        mask.save(mask_path)

        mask = face_recognition.load_image_file(mask_path)
        pil_image = Image.fromarray(mask)
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Lips
        d.polygon(face_landmarks['top_lip'], fill=(0, 0, 0, 255))
        d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['top_lip'], fill=(00, 0, 0, 255), width=1)
        d.line(face_landmarks['bottom_lip'], fill=(0, 0, 0, 255), width=1)
        # Eyebrow
        d.polygon(face_landmarks['left_eyebrow'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['left_eyebrow'], fill=(0, 0, 0, 255), width=1)
        d.polygon(face_landmarks['right_eyebrow'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['right_eyebrow'], fill=(0, 0, 0, 255), width=1)
        # Eyes
        d.polygon(face_landmarks['left_eye'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['left_eye'], fill=(0, 0, 0, 255), width=1)
        d.polygon(face_landmarks['right_eye'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['right_eye'], fill=(0, 0, 0, 255), width=1)
        # Nose
        d.polygon(face_landmarks['nose_tip'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['nose_tip'], fill=(0, 0, 0, 255), width=1)
        d.polygon(face_landmarks['nose_bridge'], fill=(0, 0, 0, 255))
        d.line(face_landmarks['nose_bridge'], fill=(0, 0, 0, 255), width=1)
        # chin
        d.polygon(face_landmarks['chin'], fill=(255, 255, 255, 0))
        d.line(face_landmarks['chin'], fill=(0, 0, 0, 255), width=3)

    # pil_image.show()
    pil_image.save(output_dir + 'mask_' + image_name_with_fe)
    shutil.copy2(load_image_path, output_dir + load_image_path.partition('/')[2])
    os.remove(mask_path)
