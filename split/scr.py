
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


def vector_coords(pt2, pt1):
    x = pt2[0][0]-pt1[0][0]
    y = pt2[0][1]-pt1[0][1]
    return x, y

def avg_vec_via_pts(v1pt2, v1pt1, v2pt2, v2pt1):
    v1 = vector_coords(v1pt2, v1pt1)
    v2 = vector_coords(v2pt2, v2pt1)
    return avg_vec(v1, v2)

def avg_vec(v1, v2):
    return 0.5*(v1[0] + v2[0]), 0.5*(v1[1] + v2[1])

def norm_vec(v):
    mod = math.sqrt(v[0]*v[0] + v[1]*v[1])
    return v[0]/mod, v[1]/mod

def vector_length(pt2, pt1):
    vx, vy = vector_coords(pt2, pt1)
    return math.sqrt(math.pow(vx, 2.0) + math.pow(vy, 2.0))


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


def rotate_and_crop(img, rectangle, indent):
    basic_pt = rectangle[0]
    basic_dx = max([vector_length(rectangle[1], rectangle[0]), vector_length(rectangle[3], rectangle[2])])
    basic_dy = max([vector_length(rectangle[3], rectangle[0]), vector_length(rectangle[2], rectangle[1])])
    rectangle_str_repr = re.sub('[\t\r\n]+', ', ', str(rectangle))
    print(
        f'rectangle_str_repr: {rectangle_str_repr};\n    rectangle dx: {basic_dx};\n    rectangle dy: {basic_dy}')
    vx = norm_vec(avg_vec_via_pts(rectangle[1], rectangle[0], rectangle[2], rectangle[3]))
    vy = norm_vec(avg_vec_via_pts(rectangle[3], rectangle[0], rectangle[2], rectangle[1]))
    vy_to_x = vy[1], -vy[0]
    x_axis = avg_vec(vx, vy_to_x)
    rotation_angle = 0.
    if abs(x_axis[0]) > 0.0000001:
        rotation_angle = math.atan(x_axis[1]/x_axis[0])/math.pi*180.
    print(f'   vx: {vx}; vy: {vy}; x_axis: {x_axis}; rotation_angle: {rotation_angle}')
    rows, cols, color = img.shape
    base_pt = rectangle[0][0][0], rectangle[0][0][1]
    M = cv.getRotationMatrix2D(base_pt,rotation_angle,1)
    dst = cv.warpAffine(img,M,(cols, rows))
    cropped = dst[base_pt[1]-indent:int(base_pt[1]+basic_dy)+indent+1, base_pt[0]-indent:int(base_pt[0]+basic_dx)+indent+1]
    return cropped


test_img_filename = get_test_filename()
dot_index = test_img_filename.rfind('.')
file_basename = test_img_filename[0:dot_index]
file_ext = test_img_filename[dot_index:]
print(f'  basename = {file_basename}; file_ext = {file_ext}')
img = get_image(test_img_filename)
threshold = get_threshold(img, 220)
rectangles = get_rectangles(threshold)
print(f'rectangles number: {len(rectangles)}')
for i in range(1, len(rectangles)):
    rectangle = rectangles[i]
    set_top_left_point_index0(rectangle)
    res = rotate_and_crop(img, rectangle, 5)
    target = file_basename + str(i) + file_ext
    cv.imwrite(target, res)

for i in range(1, len(rectangles)):
    rectangle = rectangles[i]
    color = list(np.random.random(size=3) * 255)
    cv.drawContours(img, [rectangle], -1, color, 5)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(threshold, 'gray')
ax2.imshow(img)
plt.xticks([])
plt.yticks([])

plt.show()
