
import os
import cv2 as cv
from matplotlib import pyplot as plt
import re
import numpy as np


def get_test_filename():
    print("OpenCV Version: {}".format(cv.__version__))
    print(os.path.dirname(os.path.realpath(__file__)))
    working_dir = os.path.dirname(os.path.realpath(__file__))
    img_filepath = os.path.join(working_dir, 'resources/scan_01.jpg')
    return img_filepath


def get_image(img_filename):
    img = cv.imread(img_filename)
    img = cv.copyMakeBorder(img, 50, 50, 50, 50,
                            cv.BORDER_CONSTANT, None, (255, 255, 255))
    return img


def get_threshold(img, threshold_value):
    imgGry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th1 = cv.threshold(imgGry, threshold_value, 255, cv.THRESH_BINARY)
    return th1

def get_rectangles(threshold):
    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    height, width = threshold.shape
    rectangles = []
    for contour in contours:
        bbox = cv.boundingRect(contour)
        x, y, w, h = bbox
        if w < width * 0.01 or h < height * 0.01:
            continue
        approx = cv.approxPolyDP(
            contour, 0.01 * cv.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        rectangles.append(approx)
    return rectangles


test_img_filename = get_test_filename()
img = get_image(test_img_filename)
threshold = get_threshold(img, 220)
rectangles = get_rectangles(threshold)
print(f'rectangles number: {len(rectangles)}')
for rectangle in rectangles:
    color = list(np.random.random(size=3) * 255)
    cv.drawContours(img, [rectangle], -1, color, 5)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(threshold, 'gray')
ax2.imshow(img)
plt.xticks([])
plt.yticks([])

plt.show()
