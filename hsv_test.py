import cv2
import numpy as np
from random import randrange
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 30,20

def nothing(x):
    pass

if __name__ == '__main__':
    image = cv2.imread('images/bad2.jpg')
    img = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    diff = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    diff = cv2.resize(diff, (300, 500))

    highH,highS,highV = 179,255,255
    lowH,lowS,lowV = 0, 0, 0

    mask = cv2.inRange(diff,  (lowH,lowS,lowV), (highH,highS,highV))
    cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("highH", 'mask', 0, 179, nothing)
    cv2.createTrackbar("highS", 'mask', 0, 255, nothing)
    cv2.createTrackbar("highV", 'mask', 0, 255, nothing)

    cv2.createTrackbar("lowH", 'mask', 0, 179, nothing)
    cv2.createTrackbar("lowS", 'mask', 0, 255, nothing)
    cv2.createTrackbar("lowV", 'mask', 0, 255, nothing)

    while True:
        highH = cv2.getTrackbarPos("highH", "mask")
        highS = cv2.getTrackbarPos("highS", "mask")
        highV = cv2.getTrackbarPos("highV", "mask")

        lowH = cv2.getTrackbarPos("lowH", "mask")
        lowS = cv2.getTrackbarPos("lowS", "mask")
        lowV = cv2.getTrackbarPos("lowV", "mask")

        mask = cv2.inRange(diff, (lowH, lowS, lowV), (highH, highS, highV))
        cv2.imshow("mask", mask)
        if(cv2.waitKey(30) == 27):
            break

    cv2.destroyAllWindows()
