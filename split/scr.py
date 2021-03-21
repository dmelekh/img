
import os
import cv2 as cv
from matplotlib import pyplot as plt
import re
import numpy as np
import math


def get_test_filename():
    print("OpenCV Version: {}".format(cv.__version__))
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


def r2(pt):
    return math.pow(pt[0][0], 2.0) + math.pow(pt[0][1], 2.0)


def vector_length(pt2, pt1):
    return math.sqrt(math.pow(pt2[0][0]-pt1[0][0], 2.0) + math.pow(pt2[0][1]-pt1[0][1], 2.0))


def set_top_left_point_index0(rectangle):
    r2_lst = []
    for i in range(len(rectangle)):
        r2_lst.append(r2(rectangle[i]))
    min_i = r2_lst.index(min(r2_lst))
    if min_i == 0:
        return
    for i in range(len(rectangle)-min_i):
        buffer = np.copy(rectangle[len(rectangle)-1])
        for j in range(len(rectangle)-1, 0, -1):
            rectangle[j] = rectangle[j-1]
        rectangle[0] = buffer


def rotate_and_crop(img, rectangle):
    basic_pt = rectangle[0]
    basic_di = 0.5 * \
        (vector_length(rectangle[3], rectangle[0]) +
         vector_length(rectangle[2], rectangle[1]))
    basic_dj = 0.5 * \
        (vector_length(rectangle[1], rectangle[0]) +
         vector_length(rectangle[3], rectangle[2]))
    rectangle_str_repr = re.sub('[\t\r\n]+', ', ', str(rectangle))
    print(
        f'rectangle_str_repr: {rectangle_str_repr};\n    rectangle side i: {basic_di};\n    rectangle side j: {basic_dj}')


test_img_filename = get_test_filename()
img = get_image(test_img_filename)
threshold = get_threshold(img, 220)
rectangles = get_rectangles(threshold)
print(f'rectangles number: {len(rectangles)}')
for rectangle in rectangles:
    set_top_left_point_index0(rectangle)
    color = list(np.random.random(size=3) * 255)
    cv.drawContours(img, [rectangle], -1, color, 5)
    # rotate_and_crop(img, rectangle)
    


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(threshold, 'gray')
ax2.imshow(img)
plt.xticks([])
plt.yticks([])

plt.show()
