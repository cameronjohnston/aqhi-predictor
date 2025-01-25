import configparser
import os


def load_config():
    config = configparser.ConfigParser()

    # Get the directory above the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(base_dir, '..', 'config.ini')

    # Normalize the path
    config_file_path = os.path.abspath(config_file_path)

    config.read(config_file_path)
    return config
