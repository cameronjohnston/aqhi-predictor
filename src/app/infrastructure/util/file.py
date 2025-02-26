
import logging
import os
import re
import time


def rotate_file(folder_name, file_name):
    """
    Rotate a file, ex., file.log -> file.0.log or file.1.log -> file.2.log

    :param folder_name: The folder in which to find the file
    :param file_name: The name of the file to rotate
    :return: None
    """
    logging.info(f'Rotating for {folder_name} {file_name}')
    # Make sure file has the format <name>.<extension>
    match = re.match(r'(.+)\.([^\.]+)', file_name)
    if match:
        # Use regex match to split file name into base and extension
        base_name, extension = match.groups()
        # If file doesn't exist, we don't need to do anything. If it does we need to rotate it
        src_path = os.path.join(folder_name, file_name)
        if os.path.isfile(src_path):
            # Find an empty spot for the file
            rotation = 0
            while True:
                dst_name = '{}.{}.{}'.format(base_name, rotation, extension)
                dst_path = os.path.join(folder_name, dst_name)
                if os.path.isfile(dst_path):
                    # Keep searching
                    rotation += 1
                else:
                    # We found a slot. Rotate and break loop
                    time.sleep(1)
                    os.rename(src_path, dst_path)
                    break
    else:
        msg = 'Filename %s does not match expected pattern: <name>.<extension>' % file_name
        raise RuntimeError(msg)


def prepare_file_path(full_path, rotate=True):
    """
    Prepares path to file. Makes necessary directories and rotates out existing file at that
    location if it exists and rotate set to True

    :param full_path: The full path to file
    :param rotate: Whether or not to rotate existing files
    :return: Full path we can write to
    """
    folder_name = os.path.dirname(full_path)
    file_name = os.path.basename(full_path)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if rotate:
        rotate_file(folder_name, file_name)

    return full_path


def prepare_dated_file_path(folder_name, date, file_name, rotate=True):
    """
    Prepares a folder for writing a new file. Given folder_name, date, and file_name will prepare
    a directory, folder_name/YYMM/DD/, if does not exist and rotate out existing file if exists

    :param folder_name: Path to base folder
    :param date: The date of the file. Will be used to create path
    :param file_name: The name of the file
    :param rotate: Whether or not to rotate existing files
    :return: Full path we can write to
    """

    date_str = date.strftime('%Y%m')
    day_str = date.strftime('%d')

    full_path = os.path.join(folder_name, date_str, day_str, file_name)

    return prepare_file_path(full_path, rotate)


