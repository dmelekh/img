import os


def get_folder(message):
    check_path = os.path.isdir
    folder = get_fs_element('folder', check_path, message)
    return folder

def get_folders(message):
    check_path = os.path.isdir
    folder = get_fs_elements('folder', check_path, message)
    return folder

def get_file(message):
    check_path = os.path.isfile
    return get_fs_element('file', check_path, message)

def get_files(message):
    check_path = os.path.isfile
    return get_fs_elements('file', check_path, message)

def get_fs_element(fs_elem_name, check_path, message):
    print(message)
    uIn = get_ui()
    if (uIn == ''):
        return
    if check_path(uIn):
        return uIn
    else:
        another_message = '{}\n{} does not exist, try another {}'.format(uIn, fs_elem_name, fs_elem_name)
        return get_fs_element(fs_elem_name, check_path, another_message)

def get_fs_elements(fs_elem_name, check_path, message):
    print(message)
    uIns = get_ui_list()
    if (uIns == []):
        return
    for uIn in uIns:
        if not check_path(uIn):
            another_message = '{}\n{} does not exist, try another {}'.format(uIn, fs_elem_name, fs_elem_name)
            get_fs_elements(fs_elem_name, check_path, another_message)
    return uIns

def get_ui():
    uIn = input().lstrip().rstrip()
    return uIn

def get_ui_list():
    uin_list = []
    while True:
        uIn = input().lstrip().rstrip()
        if uIn == '' or uIn is None:
            break
        uin_list.append(uIn)
    return uin_list

def get_directory_own_subdirs_abspath(directory):
    return get_directory_contents(os.path.isdir, path_fun_abspath, directory)

def get_directory_own_subdirs_basename(directory):
    return get_directory_contents(os.path.isdir, path_fun_basename, directory)

def get_directory_content_files_abspath(directory, extensions=None):
    return get_directory_contents(os.path.isfile, path_fun_abspath, directory, extensions)

def get_directory_content_files_basename(directory, extensions=None):
    return get_directory_contents(os.path.isfile, path_fun_basename, directory, extensions)

def get_directory_contents(fs_elem_fun, path_fun, directory, extensions=None):
    check_is_dir(directory)
    contents = os.listdir(directory)
    fs_elem = []
    for i in range(len(contents)):
        abspath = path_fun_abspath(directory, directory, contents[i])
        name_to_add = path_fun(directory, directory, contents[i])
        if fs_elem_fun(abspath):
            if extensions is None:
                fs_elem.append(name_to_add)
            else:
                if os.path.splitext(abspath)[1] in extensions:
                    fs_elem.append(name_to_add)
    return fs_elem

def get_dir_and_subdirs_content_files_abspath(directory, extensions=None):
    return get_dir_and_subdirs_files(path_fun_abspath, directory, extensions)

def get_dir_and_subdirs_content_files_basename(directory, extensions=None):
    return get_dir_and_subdirs_files(path_fun_basename, directory, extensions)

def get_dir_and_subdirs_files(path_fun, dir_abspath, extensions=None):
    selected_files = []
    for root, dirs, files in os.walk(dir_abspath):
        for file in files:
            if extensions is None or os.path.splitext(file)[1] in extensions:
                selected_files.append(path_fun(dir_abspath, root, file))
    return selected_files

def get_subdirs_abspath(directory):
    return get_subdirs(path_fun_abspath, directory)

def get_subdirs_basename(directory):
    return get_subdirs(path_fun_basename, directory)

def get_subdirs_relpath(directory):
    return get_subdirs(path_fun_relpath, directory)

def get_subdirs(path_fun, dir_abspath):
    selected_files = []
    for root, dirs, files in os.walk(dir_abspath):
        for dir in dirs:
            selected_files.append(path_fun(dir_abspath, root, dir))
    return selected_files

def check_is_dir(directory):
    '''
    if not a directory throws NotADirectoryError
    :param directory:
    :return:
    '''
    if not os.path.isdir(directory):
        raise NotADirectoryError(directory)

def path_fun_abspath(root, dir, basename):
    return os.path.join(dir, basename)

def path_fun_basename(root, dir, basename):
    return basename

def path_fun_relpath(root, dir, basename):
    return os.path.relpath(os.path.join(dir, basename), root)

def filter_extensions(files, extensions):
    filtered = []
    for file in files:
        for extension in extensions:
            if file[-len(extension):] == extension:
                filtered.append(file)
    return filtered

def insert_newlines_symbols_to_string_lines(lines, newline ='\n'):
    for i in range(len(lines)):
        lines[i] = lines[i] + newline
    return lines

def append_newlines_symbols_to_list_lines(lines, newline ='\n'):
    for line in lines:
        line.append(newline)
    return lines
