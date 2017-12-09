import numpy as np
import cv2, os, glob, pickle, logging, time, ulti
import _pickle as cPickle
from scipy.cluster.vq import vq 
from sklearn import svm, metrics
from algorithms import kernels, multi_classifier
def cal_histogram_of_words(image, vocabulary):
   """
   Calculate histogram of words for the image as its feature using for svm

   Parameters:
   -----------
   image : numpy array (2D)
       the gray scale image

   vocabulary : numpy array (2D)
       each row is a words (128D SIFT vector, center of cluster when running kmeans for bow)

   Returns:
   --------
   hist : numpy array
       histogram of words for the image
   """

   n_words = vocabulary.shape[0]
   hist = np.zeros(n_words)
   sift = cv2.xfeatures2d.SIFT_create()
   kpoints, descriptors = sift.detectAndCompute(image,None)
   words, distance = vq(descriptors,vocabulary)
   for word in words:
       hist[word] += 1
   return hist

def cal_features_for_images(directory):
    """
    Calculate features for all images in directory

    Feature is histogram of words.

    Vocabulary is loaded from "../bow/vocabulary.pkl".

    Vocabulary has 3500 words.

    n_images from test folder is 318 image.

    X_test : features of test images, saved in "features/X_test.pkl"
    y_test : labels of test images, saved in "features/y_test.pkl"
    filenames will be saved in "src/log/filenames.log" for access after run svm

    Returns:
    --------
    0 if success
    """

    X_test = []
    y_test = []
    directory = '..\\datasets\\test\\*'
    log_filenames = open('log\\filenames.log', 'w')
    with open('..\\bow\\vocabulary.pkl', 'rb') as fid:
        vocabulary = pickle.load(fid)
    print('n_words: {}'.format(vocabulary.shape[0]))
    for folder in glob.glob(directory):
        drive, path = os.path.splitdrive(folder)
        path, label = os.path.split(path)
        for filename in glob.glob(folder+'/*.*'):
            print('calculating for imgage {}'.format(filename))
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            hist = cal_histogram_of_words(im, vocabulary)
            #print('hist: {}'.format(hist))
            X_test.append(hist)
            y_test.append(label)
            log_filenames.write('{} \n'.format(filename))
    with open('..\\features\\X_test.pkl', 'wb') as fp:
        pickle.dump(X_test, fp)
    with open('..\\features\\y_test.pkl', 'wb') as fp:
        pickle.dump(y_test, fp)
    return 0

if __name__ == '__main__':

    print('Loading X_train, y_train')
    with open('..\\features\\X_train.pkl', 'rb') as fid:
        X_train = pickle.load(fid)
    with open('..\\features\\y_train.pkl', 'rb') as fid:
        y_train = pickle.load(fid)
    print('X_train: {}', np.shape(X_train))
    print('y_train: {}', np.shape(y_train))

    #cal_features_for_test_images()
    with open('..\\features\\X_test.pkl', 'rb') as fid:
        X_test = pickle.load(fid)
    with open('..\\features\\y_test.pkl', 'rb') as fid:
        y_test = pickle.load(fid)
    print('X_test: {}', np.shape(X_test))
    print('y_test: {}', np.shape(y_test))

    #kernel = 'linear'
    #C = 0.5
    #multi_class = 'ovr'
    #model = svm.LinearSVC(C = C)
    
    #kernel = 'rbf'
    #gamma = 0.1
    #C = 1
    #multi_class = 'ovo'
    #model = svm.SVC(kernel = kernel, gamma = gamma, C = C)

    kernel = kernels.LinearKernel()
    model = multi_classifier.OneVsOneClassifier()
    C = 1.0

    logger = ulti.create_logger('log\\01_C_{}_{}_{}.log'.format(C, kernel.getName(), model.toString()), logging.DEBUG, logging.DEBUG)
    logger.info('*'*100)
    logger.info('C: {}'.format(C))
    logger.info('kernel: {}'.format(kernel.toString()))

    start_time = time.time()
    model.fit(X_train, y_train, kernel = kernel, C = C)
    fit_time = time.time() - start_time
    logger.info('fit_time: {} seconds'.format(fit_time))
    model_file = 'bin\\models\\01_C_{}_{}_{}.pkl'.format(C, kernel, multi_class)
    with open(model_file, 'wb') as fid:
        cPickle.dump(model, fid) 
    logger.info('model is saved in {}'.format(model_file))

    y_pred = model.predict(X_test)
    logger.info('classification report:\n {}'.format(metrics.classification_report(y_test, y_pred)))
    logger.info('score: {}'.format(model.score(X_test, y_test)))
