import numpy as np
import cv2                #影象處理庫OpenCV
import dlib               #人臉識別庫dlib
#dlib預測器
detector = dlib.get_frontal_face_detector()    #使用dlib庫提供的人臉提取器
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')   #構建特徵提取器

# cv2讀取影象
img = cv2.imread("input/Tzu-yu.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人臉數rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])  #人臉關鍵點識別
    for idx, point in enumerate(landmarks):        #enumerate函式遍歷序列中的元素及它們的下標
        # 68點的座標
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)

        # 利用cv2.circle給每個特徵點畫一個圈，共68個
        cv2.circle(img, pos, 5, color=(0, 255, 0))
        # 利用cv2.putText輸出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        #各引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
        cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)

cv2.namedWindow("img", 2)
cv2.imshow("img", img)       #顯示影象
cv2.waitKey(0)        #等待按鍵，隨後退出