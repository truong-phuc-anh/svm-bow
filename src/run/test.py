import numpy as np
import cv2
import os, glob
def calculate_accuracy(y, y_predit):
    mis_indices = np.where(y != y_predit)[0]
    accuracy = 1.0 - ( 1.0 * len(mis_indices) / y.shape[0])
    return accuracy, mis_indices
def loadAllImage(strDirect):
    list_image = []
    list_label = []
    for folder in glob.glob(strDirect): #assuming gif
        drive, path = os.path.splitdrive(folder)
        path, labels = os.path.split(path)
        for filename in glob.glob(folder+'/*.*'):
            img = cv2.imread(filename)
            print(filename)
            list_image.append(img)
            list_label.append(labels)
    return list_image,list_label

if __name__ == '__main__':
    #img = cv2.imread('..\\dataset\\caltech_101\\accordion\\image_001.jpg')
    #print(img)
    #cv2.imshow('rgb', img)
    #loadAllImage('..\\dataset\\caltech_101\\*')
    file = '..\\dataset\\caltech_101\\butterfly\\image_0003.jpg'
    im = cv2.imread(file)
    cv2.imshow('file',im)
    gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    print('keypoints: ({})'.format(len(kp)))
    #print('{}'.format(kp))
    print('des: {}'.format(np.shape(des)))
    #iris = datasets.load_iris()
    #X_train = iris.data
    #y_train = iris.target
    
    #file_X = '../dataset/X_bow.pkl'
    #file_y = '../dataset/y_bow.pkl'
    #with open(file_X, 'wb') as fid:
    #    pickle.dump(X_train, fid)
    #with open(file_y, 'wb') as fid:
    #    pickle.dump(y_train, fid)

    #caltech101 = dataset.load_dataset(file_X, file_y)
    #X_train = caltech101.X
    #y_train = caltech101.y
    #print('n_samples, nfeatures: {} {}'.format(X_train.shape[0], X_train.shape[1]))
    #print('X_train:/n {}'.format(X_train))
    #print('y_train:/n {}'.format(y_train))
    #print(X_train[0][0])
    #C = 1.0  # SVM regularization parameter
    #models = (svm.SVC(kernel='linear', C=C),
    #      svm.LinearSVC(C=C),
    #      svm.SVC(kernel='rbf', gamma=0.7, C=C),
    #      svm.SVC(kernel='poly', degree=3, C=C))
    #models = (clf.fit(X_train, y_train) for clf in models)

    #for model in models:
    #    y_pred = model.predict(X_train)
    #    mis_indices, accuracy = calculate_accuracy(y_train, y_pred)
    #    print('accuracy: {}'.format(accuracy))
    #    print('mis: {}'.format(mis_indices))
    #    print('y_pred:/n {}'.format(y_pred))
