import numpy as np
from cv2 import cv2
import os
from os.path import expanduser

home = expanduser("~")
path = os.path.dirname(os.path.realpath(__file__))

def openImage(path):
    # image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)      # dua anh load vao ve mau trang den
    image = cv2.imread(path)
    return image

def rescale1(image, l = 0.1, u = 1):
    image = image.astype(np.float64)
    inmin = image.min()
    inmax = image.max()
    return l + ((image-inmin)/(inmax-inmin))*(u-l)

def rescale255(image):
    image = rescale1(image, 0, 255)
    return image.astype(np.uint8)

def resizeImage(image, newWidth):
    height, width = image.shape[:2]
    newHeight = int((newWidth * height) / width)
    return cv2.resize(image, (newWidth, newHeight), fx= 0.5, fy=0.5, interpolation= cv2.INTER_AREA)

def saveImage(name, image):
    cv2.imwrite(home + "/" + name, image)
    return home + "/" + name

def customSaveImage(path, image):
    cv2.imwrite(path, image)

