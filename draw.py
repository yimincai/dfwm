import face_recognition
import cv2
import numpy as np
from pil import Image, ImageDraw

# def create_image(shape):
#     img = np.zeros([400, 400, 3], np.uint8)+255
#     img.save('output/mask.jpg')

image_name = 'Tzu-yu.jpg'
face_line = list()
mask_path = 'output/mask.jpg'
image = face_recognition.load_image_file('input/Tzu-yu.jpg')
face_locations = face_recognition.face_locations(image, model='cnn')
top, right, bottom, left = face_locations[0]
print('A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}'.format(top, left, bottom, right))
face_image = image[top:bottom, left:right]
pil_image = Image.fromarray(face_image)
pil_image.show()
pil_image.save('output/_face_detect_' + image_name)

face_landmarks_lists = face_recognition.face_landmarks(image)
for face_landmarks in face_landmarks_lists:

    # 找出每張臉的輪廓產生臉部mask
    # top_lip = face_landmarks['top_lip']
    # bottom_lip = face_landmarks['bottom_lip']
    # left_eyebrow = face_landmarks['left_eyebrow']
    # right_eyebrow = face_landmarks['right_eyebrow']
    # left_eye = face_landmarks['left_eye']
    # right_eye = face_landmarks['right_eye']
    # nose_tip = face_landmarks['nose_tip']
    # nose_bridge = face_landmarks['nose_bridge']
    # chin = face_landmarks['chin']
    # chin.reverse()
    # list_all = top_lip + bottom_lip + left_eyebrow + right_eyebrow + left_eye + right_eye + nose_tip + nose_bridge
    # face_line.append(list_all)

    # white background
    mask = np.zeros(image.shape, np.uint8) + 255
    # black background
    # mask = np.zeros(image.shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    q = ImageDraw.Draw(mask)
    mask.save(mask_path)

    mask = face_recognition.load_image_file(mask_path)
    pil_image = Image.fromarray(mask)
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # pil_image = Image.fromarray(image)
    # d = ImageDraw.Draw(pil_image, 'RGBA')

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
    # d.polygon(face_landmarks['chin'], fill=(0, 0, 0, 255))
    # d.line(face_landmarks['chin'], fill=(0, 0, 0, 255), width=3)

pil_image.show()
pil_image.save("output/_draw_face_identification_" + image_name)


