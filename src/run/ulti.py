import logging
import numpy as np


def create_logger(file_name, file_level = None, console_level = None):
    """

    Create a custom logger using for both file and console

    Parameters
    ----------
    file_name : string
        Name of log file and also name of logger

    file_level : logging.level (optional, default = None)
        Logging level for log to file

    console_level : logging.level (optional, default = None)
        Logging level for log to console

    Returns
    -------
    logger : logger
        A logger
    """
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)
    if console_level != None:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch_format = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(ch_format)
        logger.addHandler(ch)
    if file_level != None:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh_format = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(fh_format)
        logger.addHandler(fh)
    return logger

def calculate_accuracy(y, y_predit):
    mis_indices = np.where(y != y_predit)[0]
    accuracy = 1.0 - ( 1.0 * len(mis_indices) / len(y))
    return mis_indices, accuracy