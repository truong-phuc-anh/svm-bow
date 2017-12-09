from scipy.stats import mode
from itertools import combinations
import smo
import numpy as np

# n_classifiers = n_classes * (n_classes - 1) / 2
class OneVsOneClassifier:
    def __init__(self):
        self.classifiers = []
        self.class_pairs = []
        self.name = 'ovo'

    def create_trainning_data(self, X, y):
        training_data = []
        self.class_pairs = list(combinations(set(y), 2))
        #print('class_pairs:\n {}'.format(self.class_pairs))
        for class_pair in self.class_pairs:
            # index of sample with labels in one of class_pair labels
            class_mask = np.where((y == class_pair[0]) | (y == class_pair[1]))
            # change labels of class_pair to negative and positive (to use binary classifier)
            y_i = np.where(y[class_mask] == class_pair[0], 1, -1)
            # add samples and it labels to trainning set
            training_data.append((X[class_mask], y_i))
        return training_data

    def fit(self, X, y, kernel, C):
        training_data = self.create_trainning_data(X, y)
        # Train one classifier per pair
        self.classifiers = []
        for data in training_data:
            clf = smo.SMO(kernel = kernel, C = C)
            clf.fit(data[0], data[1])
            self.classifiers.append(clf)

    def predit(self, X_unknown):
        # each row is prediction of a sample in X_unknown
        predictions = np.zeros((X_unknown.shape[0], len(self.classifiers)))
        #print('predictions:\n {}'.format(predictions))
        for idx, clf in enumerate(self.classifiers):
            class_pair = self.class_pairs[idx]
            # print('class_pair:\n {}'.format(class_pair))
            prediction = clf.predict(X_unknown)
            predictions[:, idx] = np.where(prediction == 1, class_pair[0], class_pair[1])
            # print('predictions:\n {}'.format(predictions))
        # with each row in predictions, return common value
        # example [1, 2, 1 ,4, 3, 0] -> return 1
        return np.asarray(mode(predictions, axis=1)[0].ravel().astype(int))

    def toString(self):
        return self.name

# n_classifiers = n_classes
class OneVsRestClassifier:
    def __init__(self):
        self.classifiers = []
        self.classes = []
        self.name = 'ovr'

    def fit(self, X, y, kernel, C):
        self.classes = np.unique(y)
        y_list = []
        for c in self.classes:
            y_c = np.where(y == c, 1, -1)
            y_list.append(y_c); 
        # Train one binary classifier on each problem (each class vs the rest)
        self.classifiers = []
        for y_i in y_list:
            clf = smo.SMO(kernel=kernel, C=C)
            clf.fit(X, y_i)
            print('classifier: w = {}, b = {}'.format(clf.w, clf.b))
            self.classifiers.append(clf)

    def predit(self, X):
        probs = np.zeros((X.shape[0], len(self.classifiers)))
        for idx, clf in enumerate(self.classifiers):
            probs[:, idx] = clf.get_probability_scores(X, method = 'sigmod')
        # print('probabilities:\n {}'.format(probs))
        max_indices = np.argmax(probs, axis = 1)
        return np.asarray([self.classes[idx] for idx in max_indices])

    def toString(self):
        return self.name
