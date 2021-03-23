
from common.methods import get_folder, get_file
from split.scr import split_image
import os


def split_dir():
    folder = get_folder('enter folder name')
    content = os.listdir(folder)
    print(content)
    global_index = 0
    for file in content:
        fullname = os.path.join(folder, file)
        if not os.path.isfile(fullname):
            continue
        global_index = split_image(fullname, global_index, False)

def split_file():
    file = get_file('enter file name')
    global_index = 0
    global_index = split_image(file, global_index, True)

if __name__ == "__main__" and __package__ is None:
    split_dir()
