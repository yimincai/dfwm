import face_recognition
import os
import shutil

import numpy as np
from pil import Image, ImageDraw


def create_mask(load_image_path, stored_mask_path):
    image_name = load_image_path.partition('.')[0].partition('/')[2]
    image_name_with_fe = load_image_path.partition('/')[2]
    output_dir = 'output/' + image_name + '/'
    os.makedirs(output_dir, exist_ok=True)
    image = face_recognition.load_image_file(load_image_path)
    face_locations = face_recognition.face_locations(image, model='cnn')
    top, right, bottom, left = face_locations[0]
    print('A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}'.format(top, left, bottom, right))
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
        mask.save(stored_mask_path)

        mask = face_recognition.load_image_file(stored_mask_path)
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


image_path = 'input/Barack-Obama.jpg'
mask_path = 'output/mask.jpg'
create_mask(image_path, mask_path)
