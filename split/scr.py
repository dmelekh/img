
import os
import cv2 as cv
from matplotlib import pyplot as plt

print("OpenCV Version: {}".format(cv.__version__))

print(os.path.dirname(os.path.realpath(__file__)))

working_dir = os.path.dirname(os.path.realpath(__file__))
pict_filepath = os.path.join(working_dir, 'resources/scan_01.jpg')
print(os.path.exists(pict_filepath))

img = cv.imread(pict_filepath,0)


ret,th1 = cv.threshold(img,230,255,cv.THRESH_BINARY)

with open('split/log.txt', 'w') as f:
    f.writelines([''.join(str(i))+'\n' for i in th1.tolist()])

print(len(th1), len(th1[0]))

plt.imshow(th1,'gray')
plt.xticks([])
plt.yticks([])

# plt.show()
