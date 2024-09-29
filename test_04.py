import cv2
import numpy as np
import mss

# 加载人脸检测模型 (确保路径指向正确的 XML 文件)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

# 初始化屏幕截图工具
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 从摄像头捕获帧
    ret, frame = cap.read()
    # 将截图转换为 NumPy 数组
    img = np.array(frame)

    # 将图像的颜色从 BGRA 转换为 BGR 以便 OpenCV 处理
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 转换为灰度图像以进行人脸检测
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # 遍历所有检测到的人脸，并绘制红色矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制红色矩形框

    # 显示屏幕图像
    cv2.imshow("Screen", img_bgr)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭窗口
cv2.destroyAllWindows()
