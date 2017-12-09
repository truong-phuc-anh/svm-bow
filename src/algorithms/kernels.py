import numpy as np

class LinearKernel:
    def __init__(self):
        self.name = 'Linear Kernel'
    def apply(self, x1, x2):
        return np.dot(x1, x2)
    def toString(self):
        return self.name
    def getName(self):
        return 'Linear'

class PolinomialKernel:
    def __init__(self, degree, constrant = 0):
        self.degree = degree
        self.constrant = constrant
        self.name = 'Polinomial Kernel with degree = {}, constrant = {}'.format(self.degree, self.constrant)
    def apply(self, x1, x2):
        sum = np.dot(x1, x2) + self.constrant
        return pow(sum, self.degree)
    def toString(self):
        return self.name
    def getName(self):
        return 'Polinomial'

class RBFKernel:
    def __init__(self, gamma):
        self.gamma = gamma
        self.name = 'RBF Kernel (Gaussian) with gamma = {}'.format(self.gamma)
    def apply(self, x1, x2):
        return np.exp(-self.gamma * pow(np.linalg.norm(x1 - x2), 2))
    def toString(self):
        return self.name
    def getName(self):
        return 'rbf'
