import cv2
import glob
import os
import numpy as np

def load_images(dataset_name):
    """
    
    Load images from dataset folder

    Parameters
    ----------
    dataset_name : string
        Name of dataset need to load

    Returns
    -------
    imgs : numpy.ndarray
        List of images in dataset (each row is an image in RGB)

    labels : numpy.ndarray(string)
        Labels of images in dataset

    """
    imgs = []
    labels = []
    full_path = '../dataset/{}/*'.format(dataset_name)
    for folder in glob.glob(full_path):
        drive, path = os.path.splitdrive(folder)
        path, names = os.path.split(path)
        for filename in glob.glob(folder+'/*.*'):
            img = cv2.imread(filename)
            imgs.append(img)
            labels.append(names)
    return np.array(imgs), np.array(labels)
