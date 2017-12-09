import numpy

# const
default_dimension = 2

# sumary:
#   used w to classify a sample x
# parameters:
#   x: a sample
#   w: weights
# returns:
#   label of x (classify by hyperplan wx = 0) 
def hypothesis(x, w):
    if numpy.dot(x, w) >= 0:
        return 1
    return -1
    # warning numpysign not work with 0, sign(0) = 0
    #return numpy.sign(numpy.dot(x, w))

# summary:
#   use w to predict trainning set 
# parameters:
#   hypothesis_func: function used to predict a particular sample
#   X: trainning set
#   Y_expected: expected labels
#   w: weights of hyperplan
# returns:
#   mis_samples: misclassified samples (have predicted labels != expected labels)
def predict(hypothesis_func, X, Y_expected, w):
    # apply function hypothesis for all sample in X (axis = 1 means by horizontal of X)
    Y_predicted = numpy.apply_along_axis(hypothesis_func, 1, X, w)
    # get mis_samples (with Y_predicted != Y_expected )
    mis_samples = X[Y_expected != Y_predicted]
    return mis_samples

# summary:
#   pick a random sample from a set
# parameters:
#   set: set that we will get a sample from (set is a sub-set of X)
#   X: trainning set
#   Y: expected labels
# returns:
#   x: a sample from set
#   y: expected label of x
def pick_one_from(set, X, Y):
    random_index = numpy.random.randint(len(set))
    random_sample = set[random_index]
    index = numpy.where(numpy.all(X == random_sample, axis = 1))
    y_expected = Y[index]
    return random_sample, y_expected

# summary:
#   apply 'update rule' to get a new weights which may correct a mis-sample
#   update rule: change sign of wx = change sign of ||w||.||x||.cos(a) = change sign of cos(a)
#       if 0 < a <= 90: cos(a) >= 0, we need cos(a) < 0 -> decrease a -> new_w = w + x
#       if 90 < a <= 180: cos(a) < 0, we need cos(a) > 0 -> increase a -> new_w = w - x 
# parameters:
#   w: weight before update
#   mis_sample: a misclassified sample
#   y_expected: expected label of mis_sample
# returns:
#   new_w: updated w
def apply_update_rule(w, mis_sample, y_expected):
    new_w = w + mis_sample * numpy.sign(y_expected)
    return new_w

# perceptron learning algorithm
# X: trainning set (augmented) all sample x in X has x0 = 1. x = {1, x1, x2, ...}
# Y: expected label (ground truth)
# return: w = weights of hyperplan
def pla(X, Y):
    
    # pick a random w
    w = numpy.random.rand(default_dimension + 1) # +1 means using augmented vector
    print('init w = {0}'.format(w))

    # use this w to predict on X and get misclassified samples set
    mis_samples = predict(hypothesis, X, Y, w)

    # if have any misclassified samples
    while mis_samples.any():
        print('number of mis_samples = {0}'.format(len(mis_samples)))
        # pick a random mis-sample and its expected label
        mis_sample, y_expected = pick_one_from(mis_samples, X, Y)
        # update w (try to correct the mis-sample's label, don't care other samples)
        w = apply_update_rule(w, mis_sample, y_expected)
        print('updated w = {0}'.format(w))
        # predict again
        mis_samples = predict(hypothesis, X, Y, w)

    # convergence: if trainning set X is linearly separable
    # todo how to check whether X is linearly separable or not
    return w
