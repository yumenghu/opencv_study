import logging

import cv2
import numpy as np
import mss

# 读取目标图片模板
template = cv2.imread('wangzhe.jpg', cv2.IMREAD_COLOR)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template.shape[1], template.shape[0]  # 获取模板的宽度和高度
# 设置匹配阈值
threshold = 0.8

# 初始化屏幕截图工具
with mss.mss() as sct:
    # 定义要截取的屏幕区域
    monitor = sct.monitors[2]  # 获取第一个监视器

    while True:
        # 从指定区域截取屏幕图像
        screenshot = sct.grab(monitor)
        # 将截图转换为 NumPy 数组
        img = np.array(screenshot)
        # 将图像的颜色从 BGRA 转换为 BGR，以便 OpenCV 正确显示
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        # 模板匹配
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        # 释放对应技能
        locations = np.where(result >= threshold)
        # 如果找到匹配，绘制红色矩形框
        for pt in zip(*locations[::-1]):  # 获取匹配位置
            cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  # 绘制红色矩形框
        # 显示屏幕图像
        cv2.imshow("Screen", img_bgr)
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
def matchAndRelease():
    pass

# 关闭窗口
cv2.destroyAllWindows()
