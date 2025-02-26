
import datetime
import logging
import logging.config
import os
import socket
import sys

from infrastructure.util.file import prepare_dated_file_path


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)-8s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'no_format': {
            'format': None,
            'datefmt': None,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'loggers': {
        '': {
            'handlers' : ['console'],
            'level'    : 'INFO',
            'propagate': True
        }
    }
}


def add_file_handler(logger, log_file_name, formatter_override=None):
    """
    Add the file handler to existing handlers

    Args:
    - logger (logger): logger to add file handler to
    - log_file_name (str): Full path to file name to indicate file logging
    - formatter_override (str): Optionally use this formatter, rather than the 'standard' formatter

    Returns: logger
    """
    file_handler = logging.FileHandler(log_file_name)

    # Use same format as default handler
    if logger.handlers:

        # Override if requested
        if formatter_override:
            formatter = logging.Formatter(
                fmt=LOGGING_CONFIG['formatters'][formatter_override]['format'],
                datefmt=LOGGING_CONFIG['formatters'][formatter_override]['datefmt']
            )
        else:  # default
            formatter = logger.handlers[0].formatter
        file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logging.info(f'#{os.getpid()} successfully added file handler with formatter override {formatter_override}')

    return logger


def setup_logging(log_level_override=None, base_dir=None, log_file_name=None):
    """
    Log to stdout (and possibly file) at specified log level

    Args:
    - log_level_override (str): Logging level (CRITICAL/ERROR/WARNING/INFO/DEBUG)
    - base_dir (str): Base dir before adding YYYYMM/DD
    - log_file_name (str, optional): Optional full path to file name to indicate file logging

    Returns: None
    """

    # Initialize from dict config
    logging.config.dictConfig(LOGGING_CONFIG)

    # Get the root logger
    root_logger = logging.getLogger()

    if log_level_override:
        root_logger.setLevel(log_level_override)

    if base_dir and log_file_name:
        full_path = prepare_dated_file_path(folder_name=base_dir, date=datetime.date.today(), file_name=log_file_name)
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(full_path)

        # Use same format as default handler
        if root_logger.handlers:
            formatter = root_logger.handlers[0].formatter
            file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log hostname, user, log file
        logging.info(f'Logging to {full_path}')
        logging.info(f'Running on {socket.gethostname()} as {os.getlogin()}')
        logging.info(f'Running cmd: {sys.executable} ' + ' '.join(sys.argv))


