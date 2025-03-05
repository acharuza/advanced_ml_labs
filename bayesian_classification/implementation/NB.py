from implementation.binary_classifier import BinaryClassifier
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

        def gaussian_density(x, mean, variance):
            first_term = 1 / np.sqrt(2 * np.pi * np.square(variance))
            second_term = np.exp(-np.square(x - mean) / (2 * variance))
            return first_term * second_term

        # calculate x given y probabilities
        x_given_y0 = np.prod(
            gaussian_density(Xtest, self.means0, self.variances0), axis=1
        )
        x_given_y1 = np.prod(
            gaussian_density(Xtest, self.means1, self.variances1), axis=1
        )

        # calculate y given x probability for class 1
        denominator = x_given_y0 * self.prior_proba0 + x_given_y1 * self.prior_proba1
        proba = x_given_y1 * self.prior_proba1 / denominator

        return proba

    def get_params(self):
        params = super().get_params()
        params["variances0"] = self.variances0
        params["variances1"] = self.variances1
        return params
