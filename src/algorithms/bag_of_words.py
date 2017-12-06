import numpy as np
import cv2

class BagOfWords:
    def __init__(self):
        self.vocabulary = np.empty(shape = [0, 0], dtype = float, order = 'C') # 2D-array, a row = 1 word = 1 SIFT descriptor = 1 center = 128D vector
    
    def create_vocabulary(self, image_names):
        imgs = cv2.imread(image_names)
        gray_imgs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        for img in gray_imgs:
            key_points, descriptors = sift.detectAndCompute(img,None)

        self.vocabulary = run_kmeans_on(descriptors) # get centers of cluster
    
    def cal_histogram(self, image):
        # create empty histogram
        n_words = self.vocabulary.shape[0]
        histogram = np.zero(n_words)
    
        # calculate all descriptors in image
        key_points = get_key_points(image)
        descriptors = get_descriptors(key_points)
    
        # for each descriptor, find out which word is it.
        for descriptor in descriptors:
            word_index = self.which_word(descriptor)
            histogram[word_index] = histogram[word_index] + 1
    
        return histogram
        
    def which_word(self, descriptor):
        n_words = len(self.vocabulary)
        dist = np.zero(n_words)
        for i in range(n_words):
            dist[i] = L1_distance(vocabulary[i], descriptor)
        return min_index_of(dist)