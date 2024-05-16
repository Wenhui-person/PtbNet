import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'E:\Yangwenhui\Projects\retinanet_new\images\CHNCXR_0331_1.png',-1)  # 读取图像,灰度模式

# 对比度增强
contrast_img = cv2.addWeighted(img, 1.2,np.zeros(img.shape, img.dtype), 0, 0)

# 显示结果
# cv2.imshow('Original', img)
# cv2.imshow('Contrast', contrast_img)
# cv2.waitKey(0)
img = np.hstack((img,contrast_img))
plt.imshow(img)
plt.axis('off')
plt.show()
