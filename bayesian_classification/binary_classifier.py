import numpy as np


class BinaryClassifier:

    def __init__(self):
        self.prior_proba0 = None
        self.prior_proba1 = None
        self.means0 = None
        self.means1 = None
        self.X0 = None
        self.X1 = None

    def fit(self, X, y):
        # calculate prior probabilities
        self.prior_proba1 = np.mean(y)
        self.prior_proba0 = 1 - self.prior_proba1

        # divide by class
        self.X0 = X[y == 0]
        self.X1 = X[y == 1]

        # calculate mean for class 0 and class 1
        self.means0 = self.X0.mean(axis=0)
        self.means1 = self.X1.mean(axis=0)

    def predict_proba(self, Xtest):
        pass

    def predict(self, Xtest, treshold=0.5):
        proba = self.predict_proba(Xtest)
        return (proba >= treshold).astype(int)

    def get_params(self):
        return self.prior_proba0, self.prior_proba1, self.means0, self.means1
