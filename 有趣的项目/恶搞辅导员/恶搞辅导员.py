import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def plt_show(img):
    imageRGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(imageRGB)
    plt.show()


def ImportFrontPhoto(FilePath):
    image = cv2.imread(FilePath, 0)  #后面一个参数为1时，是RGB模式
                               #为0时，灰度模式
                               #为-1时，包括alpha
    plt.savefig("huidu.jpg")
    plt_show(image)
    return image


def Resize(image):
    image_resize = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
    #这个函数的说明见Chaome收藏夹
    plt_show(image_resize)
    return image_resize


def ImageBinary(image):
    ret, image_binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    #第一个参数是图片，第二个参数是阈值，第三个参数表示高于阈值时赋予的新值
    #第四个cv2.THRESH_BINARY表示黑白，cv2.THRESH_BINARY_INV表示黑白二值反转
    #本例将像素值高于80的设为255，其余设为0
    #ret是得到的阈值值，image_binary是阈值化后的图像
    plt_show(image_binary)
    return  image_binary

def cutImage(image):  #区域自行划定
    image_roi = image[0: 140, 0: 125]
    plt_show(image_roi)
    return image_roi

ima = ImportFrontPhoto("jige.jpg")
ima = Resize(ima)
ima = ImageBinary(ima)
ima = cutImage(ima)
