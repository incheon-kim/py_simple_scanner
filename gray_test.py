import cv2
import numpy as np
from random import randrange
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 30,20

def nothing(x):
    pass

if __name__ == '__main__':
    image = cv2.imread('images/good.jpg')
    img = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    diff = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray = cv2.resize(gray, (500, 700))
    img = cv2.resize(img, (500,700))
    low, high = 0 , 255

    mask = cv2.inRange(gray,  low, high)

    cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("low", 'mask', 0, 255, nothing)
    cv2.createTrackbar("high", 'mask', 0, 255, nothing)

    while True:
        low = cv2.getTrackbarPos("low", "mask")
        high = cv2.getTrackbarPos("high", "mask")

        mask = cv2.inRange(gray, low, high)

        #dst, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #if len(contours) != 0:
         #   c = max(contours, key = cv2.contourArea)
        #for i in range(len(contours)):
            #dst = cv2.drawContours(img, contours, i, 255, 4, cv2.LINE_4)

        cv2.imshow("mask", mask)
        if(cv2.waitKey(30) == 27):
            break

    cv2.destroyAllWindows()
