from binary_classifier import BinaryClassifier
import numpy as np


class NB(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.variances0 = None
        self.variances1 = None

    def fit(self, X, y):
        # calculate prior probabilities and means
        super().fit(X, y)

        # calculate variances
        self.variances0 = np.var(self.X0, axis=0)
        self.variances1 = np.var(self.X1, axis=0)

    def predict_proba(self, Xtest):
        pass

    def get_params(self):
        return *super().get_params(), self.variances0, self.variances1
