import cv2
import numpy as np
import os

# printing options (super - optional)
cv2.namedWindow("roi", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("roi_gray", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("thresh", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("res", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("dst", cv2.WINDOW_NORMAL)


# to control OpenCV track bar
def nothing(x):
    pass


# adjust coordinates before transform
def rectify(h):
    h = h.reshape((4,2))
    print ("h", h)
    hnew = np.zeros((4,2),dtype = np.float32)
    print ("hnew", hnew)
    add = h.sum(1)
    print ("add", add)
    hnew[0] = h[np.argmin(add)]
    print (h[np.argmin(add)])
    hnew[2] = h[np.argmax(add)]
    print (h[np.argmax(add)])

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


# get ROI that specify where a white paper is
def get_ROI(x):
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    sensitivity = 78

    # Binarization
    mask = cv2.inRange(hsv, (0, 0, 255 - sensitivity), (255, 255 - sensitivity, 255))

    # find contours on binarized image
    dst, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest contour area
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)

    # get Points of ROI
    x, y, w, h = cv2.boundingRect(c)
    x -= 20
    y -= 20
    w += 100
    h += 100

    # return ROI image
    return  img[y:y + h, x:x + w].copy()


# find contour.
# x : mask image
# roi : original ROI
# kernel : morphology kernel
def get_target_contour(x, roi, kernel):
    x = cv2.erode(x, kernel, None, None, 3)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, None, None, 3)
    x = cv2.dilate(x, kernel,None, None, 3)

    dst, contours, hierarchy = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    dst = cv2.drawContours(roi.copy(), contours, 0, 255, 10, cv2.LINE_4)

    # print found and drawn contour
    cv2.imshow("res", dst)

    return contours


if __name__ == '__main__':
    # load image and resize into half size
    image = cv2.imread('images/bad4.jpg', cv2.IMREAD_COLOR)
    height, width = image.shape[0]//2, image.shape[1]//2
    image = cv2.resize(image, (width, height))

    # make a copy of image
    img = image.copy()

    # create kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # convert to HSV colorspace to detect a white paper
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get ROI of a white paper in Image
    roi = get_ROI(image)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # threshold ROI grayscale image
    high_thres_val, _ = cv2.threshold(roi_gray,
                                      0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    low_thres_val = high_thres_val * 0.5

    # create mask for ROI image
    _2 = _.copy()
    cv2.floodFill(_2, None, (0,0), 255)
    cv2.bitwise_xor(_, _2, _)
    cv2.bitwise_not(_, _)

    # find contour of mask
    contours = get_target_contour(_,roi_gray,kernel)

    # get 4 points of found contour
    p = cv2.arcLength(contours[0], True)

    target = cv2.approxPolyDP(contours[0], 0.02 * p, True)
    approx = rectify(target)

    # get perspective points of wanted area
    pts2 = np.float32([[0, 0], [height, 0], [height, width], [0, width]])
    target = np.float32(target)
    M = cv2.getPerspectiveTransform(target, pts2)
    dst = cv2.warpPerspective(roi, M, (height, width))

    # to draw rectangle on wanted area. It will show which area will be scanned
    cv2.drawContours(roi, [np.int32(target)], -1, (0, 255, 0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # adjust image to get better result
    th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.rotate(th3,cv2.ROTATE_90_CLOCKWISE)
    th3 = cv2.flip(th3, 1)

    # show images
    cv2.imshow("roi", roi)
    cv2.imshow("roi_gray", roi_gray)
    cv2.imshow("thresh", _)
    cv2.imshow("dst", th3)

    # save results
    if not os.path.exists("results"):
        os.mkdir("results")
    cv2.imwrite("results/roi.png", roi)
    cv2.imwrite("results/roi_gray.png", roi_gray)
    cv2.imwrite("results/threshold.png", _)
    cv2.imwrite("results/result.png", th3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
