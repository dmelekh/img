
import os
import cv2 as cv
from matplotlib import pyplot as plt
import re
import numpy as np

print("OpenCV Version: {}".format(cv.__version__))

print(os.path.dirname(os.path.realpath(__file__)))

working_dir = os.path.dirname(os.path.realpath(__file__))
pict_filepath = os.path.join(working_dir, 'resources/scan_01.jpg')
print(os.path.exists(pict_filepath))

img = cv.imread(pict_filepath)
img = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_CONSTANT, None, (255,255,255))
print(f'class name of imread() result: {img.__class__.__name__}')
imgGry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
height, width = imgGry.shape
print(f'image width: {width}')
print(f'image height: {height}')

print(width*0.1)

ret, th1 = cv.threshold(imgGry, 220, 255, cv.THRESH_BINARY)
print(f'class name of threshold() result: {th1.__class__.__name__}')
contours, hierarchy = cv.findContours(
    th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# th1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# print(f'contours len: {len(contours)}')
# for i in range(5):
#     contour_str_repr = re.sub('[\t\r\n]+', ', ', str(contours[i]))
#     print(f'contour #{i}: {contour_str_repr}')


# with open('split/log.txt', 'w') as f:
#     f.writelines([''.join(str(i))+'\n' for i in th1.tolist()])

# print(len(th1), len(th1[0]))

for contour in contours:
    bbox = cv.boundingRect(contour)
    x, y, w, h = bbox
    if w < width * 0.01 or h < height * 0.01:
        continue
    # print(f'bounding box: {bbox}')

    approx = cv.approxPolyDP(
        contour, 0.01 * cv.arcLength(contour, True), True)
    if len(approx) != 4:
        continue
    approx_str_repr = re.sub('[\t\r\n]+', ', ', str(approx))
    print(f'approx: {approx_str_repr}')
    color = list(np.random.random(size=3) * 255)
    print(f'color: {color}')
    cv.drawContours(img, [approx], -1, color, 5)
    # cv.drawContours(img, [approx], -1, (0, 255, 0), 5)
    # cv.drawContours(th1, approx, -1, (255, 0, 0), 20)

# cv.drawContours(th1, contours, -1, (255, 0, 0), 5)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(th1, 'gray')
ax2.imshow(img)
plt.xticks([])
plt.yticks([])

plt.show()
