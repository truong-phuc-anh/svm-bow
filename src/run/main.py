def get_images(dataset_name):

def get_label(img_name, dataset_name)

if __name__ == '__main__':
    # 1. create vocabulary
    images = get_images('caltech_101')
    bow = BagOfWords()
    bow.create_vocabulary(images)
    
    # 2. create dataset with features (histogram of words)
    dataset = DataSet()
    for image in images:
        dataset.X[i] = bow.cal_histogram(image) # a vector 
        dataset.y[i] = get_label(img_name, dataset_name)
    dataset.split(train_rate = 0.8, test_rate = 0.2)
    X_train, y_train = dataset.get_trainning_set()
    X_test, y_test = dataset.get_testing_set()
    
    # 3. build model using svm
    model = svm.SVC(c=1, gamma=0.001)
    model.fit(X_train, y_train)
    
    # 4. predit on trainning data
    y_predit_train = model.predit(X_train)
    accuracy_train = cal_accuracy(y_predit_train, y_train)
    
    # 5. predit on testing data
    y_predit_test = model.predit(X_test)
    accuracy_test = cal_accuracy(y_predit_test, y_test)
    
    # 6. visualize what ever you want
    