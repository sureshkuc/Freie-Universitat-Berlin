# code heavily inspired by:
# https://github.com/bobbyrathoree/Soft-Margin-SVM-with-CVXOPT/blob/master/SVM%20soft%20margins%20in%20kernels%20with%20CVXOPT.ipynb

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers


class LinearSoftMargin(object):
    
    def __init__(self, C=1.0):
        self.C = C
        self.C = float(self.C)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype('float')
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.dot(X[i], X[j])
        
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print('{0} support vectors out of {1} points'.format(len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector

        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import matplotlib.pyplot as pl
    from sklearn import datasets
    iris = datasets.load_iris()

    # transform problem into binary problem
    X, y = iris['data'], iris['target']
    for i in range(len(y)):
        if y[i] == 2:
            y[i] = 1

    # shuffle data
    idx = np.random.permutation(len(y))
    X,y = X[idx], y[idx]

    # split into train and test
    split = int(np.floor(len(y)*0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # train classifier
    clf = LinearSoftMargin(C=1)
    clf.fit(X_train, y_train)
    
    # predict and result
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print('{0} out of {1} predictions correct'.format(correct, len(y_predict)))
    
# The result is not great. Likely because there are a lot more
# of class 1 than of class 0. This causes an imbalance in the dataset.


''' Excercise 3: To make a multi-class classifier
we could simply make a one vs the rest classification for all
the classes. E.g. for four classes: 1 vs 2-3-4, then 2 vs 3-4 and
finally 3 vs 4.
'''
