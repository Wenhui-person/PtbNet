# 测试通过去除边缘像素得到的图像是什么样
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img1 = cv.imread(r"E:\Yangwenhui\Data\FJH2022\Aug\images\20150511000352-11.jpg")
    h, w, _ = img1.shape   #
    img2 = img1[50:-50, 50:-50, :]
    print(img2.shape)
    plt.imshow(img1)
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
