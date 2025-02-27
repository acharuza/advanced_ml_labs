import numpy as np


class BinaryClassifier:

    def __init__(self):
        self.prior_proba0 = None
        self.prior_proba1 = None

    def fit(self, X, y):
        self.prior_proba1 = np.mean(y)
        self.prior_proba0 = 1 - self.prior_proba1

    def predict_proba(self, Xtest):
        pass

    def predict(self, Xtest):
        pass

    def get_params(self):
        return self.prior_proba0, self.prior_proba1
