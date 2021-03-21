
from common.methods import get_folder
from split.scr import split_image
import os

if __name__ == "__main__" and __package__ is None:
    folder = get_folder('enter folder name')
    content = os.listdir(folder)
    print(content)
    global_index = 0
    for file in content:
        fullname = os.path.join(folder, file)
        if not os.path.isfile(fullname):
            continue
        global_index = split_image(fullname, global_index, False)
