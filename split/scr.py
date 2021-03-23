
import os
import cv2 as cv
from matplotlib import pyplot as plt
import re
import numpy as np
import math


def np_arr_to_string(rectangle):
    return re.sub('[\t\r\n]+', ', ', str(rectangle))


def r2(pt):
    return math.pow(pt[0][0], 2.0) + math.pow(pt[0][1], 2.0)


def vector_coords(pt2, pt1):
    x = pt2[0][0]-pt1[0][0]
    y = pt2[0][1]-pt1[0][1]
    return x, y


def vec_avg_via_pts(v1pt2, v1pt1, v2pt2, v2pt1):
    v1 = vector_coords(v1pt2, v1pt1)
    v2 = vector_coords(v2pt2, v2pt1)
    return vec_avg(v1, v2)


def vec_avg(v1, v2):
    return 0.5*(v1[0] + v2[0]), 0.5*(v1[1] + v2[1])


def vec_sum_via_pts(v1pt2, v1pt1, v2pt2, v2pt1):
    v1 = vector_coords(v1pt2, v1pt1)
    v2 = vector_coords(v2pt2, v2pt1)
    return vec_sum(v1, v2)


def vec_sum(v1, v2):
    return (v1[0] + v2[0]), (v1[1] + v2[1])


def vec_norm(v):
    mod = math.sqrt(v[0]*v[0] + v[1]*v[1])
    return v[0]/mod, v[1]/mod


def vector_length(pt2, pt1):
    vx, vy = vector_coords(pt2, pt1)
    return math.sqrt(math.pow(vx, 2.0) + math.pow(vy, 2.0))


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

    # ret, th1 = cv.threshold(imgGry, threshold_value, 255, cv.THRESH_TOZERO)

    # ret, th1 = cv.threshold(imgGry, threshold_value, 255, cv.THRESH_TRUNC)

    # ret, th1 = cv.threshold(imgGry, threshold_value, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # blur = cv.GaussianBlur(imgGry,(5,5),0)
    # ret, th1 = cv.threshold(blur, threshold_value, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th1


def avg(v1, v2):
    return (v2+v1)/2.


def mirror_point_by_line(line_pt1, line_pt2, src_pt):
    avg_x = avg(line_pt1[0][0], line_pt2[0][0])
    avg_y = avg(line_pt1[0][1], line_pt2[0][1])
    avg_pt = [[avg_x, avg_y]]
    v = vector_coords(avg_pt, src_pt)
    return [avg_pt[0][0]+v[0], avg_pt[0][1]+v[1]]


def process_triangle(triangle):
    # print(f'!!! process_triangle: {np_arr_to_string(triangle)}')
    v = [
        vector_length(triangle[1], triangle[0]),
        vector_length(triangle[2], triangle[1]),
        vector_length(triangle[0], triangle[2])
    ]
    # print(f'!!! process_triangle v: {v}')
    i_max = v.index(max(v))
    if (i_max == 0):
        m = mirror_point_by_line(triangle[1], triangle[0], triangle[2])
        return np.insert(triangle, 1, m, axis=0)
    if (i_max == 1):
        m = mirror_point_by_line(triangle[2], triangle[1], triangle[0])
        return np.insert(triangle, 2, m, axis=0)
    m = mirror_point_by_line(triangle[0], triangle[2], triangle[1])
    return np.insert(triangle, 3, m, axis=0)


def min_diagonal_is_too_short(rectangle):
    side_max = max(
        vector_length(rectangle[1], rectangle[0]),
        vector_length(rectangle[2], rectangle[1])
    )
    diagonal_min = min(
        vector_length(rectangle[2], rectangle[0]),
        vector_length(rectangle[3], rectangle[1])
    )
    return (diagonal_min <= side_max)


def process_min_diagonal(rectangle):
    diagonals = [
        vector_length(rectangle[2], rectangle[0]),
        vector_length(rectangle[3], rectangle[1])
    ]
    i_min = diagonals.index(min(diagonals))
    if i_min == 0:
        triangle = np.delete(rectangle, 2, 0)
        # print(
            # f'!!! process_min_diagonal triangle: {np_arr_to_string(triangle)}')
        return process_triangle(triangle)
    triangle = np.delete(rectangle, 3, 0)
    # print(f'!!! process_min_diagonal triangle: {np_arr_to_string(triangle)}')
    return process_triangle(triangle)


def get_rectangles(threshold, min_relative_size):
    sides_ratio_limit = 15.0
    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    height, width = threshold.shape
    rectangles = []
    for contour in contours:
        bbox = cv.boundingRect(contour)
        x, y, w, h = bbox
        if w < width * min_relative_size or h < height * min_relative_size or max(w, h) / min(w, h) > sides_ratio_limit:
            continue
        # print(bbox)
        approx = cv.approxPolyDP(
            contour, 0.01 * cv.arcLength(contour, True), True)
        if len(approx) != 4 and len(approx) != 3:
            continue
        if len(approx) == 3:
            approx = process_triangle(approx)
        if min_diagonal_is_too_short(approx):
            approx = process_min_diagonal(approx)

        rectangles.append(approx)
    return rectangles


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

def fix_contour_direction_if_needed(rectangle):
    if rectangle[1][0][1] > rectangle[3][0][1]:
        buffer = np.copy(rectangle[3])
        rectangle[3] = rectangle[1]
        rectangle[1] = buffer


def rotate_and_crop(img, rectangle, indent):
    basic_pt = rectangle[0]
    basic_dx = max([vector_length(rectangle[1], rectangle[0]),
                    vector_length(rectangle[3], rectangle[2])])
    basic_dy = max([vector_length(rectangle[3], rectangle[0]),
                    vector_length(rectangle[2], rectangle[1])])
    # print(
    #     f'rectangle_str_repr: {np_arr_to_string(rectangle)};\n    rectangle dx: {basic_dx};\n    rectangle dy: {basic_dy}')
    vx = (vec_sum_via_pts(rectangle[1],
                          rectangle[0], rectangle[2], rectangle[3]))
    vy = (vec_sum_via_pts(rectangle[3],
                          rectangle[0], rectangle[2], rectangle[1]))
    vy_to_x = vy[1], -vy[0]
    x_axis = vec_norm(vec_sum(vx, vy_to_x))

    rotation_angle = 0.
    if abs(x_axis[0]) > 0.0000001:
        rotation_angle = math.atan(x_axis[1]/x_axis[0])/math.pi*180.
    # print(f'   vx: {vx}; vy: {vy}; x_axis: {x_axis}; rotation_angle: {rotation_angle}')
    rows, cols, color = img.shape
    base_pt = rectangle[0][0][0], rectangle[0][0][1]
    M = cv.getRotationMatrix2D(base_pt, rotation_angle, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    cropped = dst[base_pt[1]-indent:int(base_pt[1]+basic_dy)+indent+1,
                  base_pt[0]-indent:int(base_pt[0]+basic_dx)+indent+1]
    return cropped


def draw_image(img, threshold, rectangles):
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


def split_image(img_filename, global_index, draw):
    dot_index = img_filename.rfind('.')
    file_basename = img_filename[0:dot_index]
    file_ext = img_filename[dot_index:]
    if not os.path.exists(file_basename):
        os.mkdir(file_basename)
    # print(f'  basename = {file_basename}; file_ext = {file_ext}')
    img = get_image(img_filename)
    rectangles_space = []
    thresholds = []
    rectangles_sum_area = []
    rectangles_len = []
    threshold_level_min = 205
    threshold_level_max = 265
    for threshold_level in range(threshold_level_min, threshold_level_max):
        threshold = get_threshold(img, threshold_level)
        rectangles = get_rectangles(threshold, 0.05)
        thresholds.append(threshold)
        rectangles_space.append(rectangles)
        rectangles_len.append(len(rectangles))
        rectangles_sum_area.append(sum(
            [vector_length(rec[1], rec[0])*vector_length(rec[2], rec[1]) for rec in rectangles]
        ))
    print(f'   rectangles_num: {rectangles_len}')
    max_rectangles_len = max(rectangles_len)
    for i in range(len(rectangles_space)):
        if (len(rectangles_space[i]) < max_rectangles_len):
            rectangles_sum_area[i] = -1.0

    i_of_max_len_and_area = rectangles_sum_area.index(max(rectangles_sum_area))
    rectangles = rectangles_space[i_of_max_len_and_area]
    threshold = thresholds[i_of_max_len_and_area]

    print(f'   i_of_max_len_and_area = {i_of_max_len_and_area}')
    print(f'   threshold_level = {threshold_level_min + i_of_max_len_and_area}')

    for i in range(1, len(rectangles)):
        rectangle = rectangles[i]
        set_top_left_point_index0(rectangle)
        fix_contour_direction_if_needed(rectangle)
        print(f'   rectangle: {np_arr_to_string(rectangle)}')
        res = rotate_and_crop(img, rectangle, 10)
        target = os.path.join(
            file_basename, '{:04d}'.format(global_index) + file_ext)
        cv.imwrite(target, res)
        global_index += 1
    if draw:
        draw_image(img, threshold, rectangles)
    return global_index
