from .binary_classifier import BinaryClassifier
import numpy as np


class LDA(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.cov_matrix = None

    def fit(self, X, y):
        # calculate prior probabilities and means
        super().fit(X, y)

        # calculate covariance matrix
        n0 = len(self.X0)
        n1 = len(self.X1)

        cov0 = np.cov(self.X0, rowvar=False)
        cov1 = np.cov(self.X1, rowvar=False)
        self.cov_matrix = (cov0 * (n0 - 1) + cov1 * (n1 - 1)) / (n0 + n1 - 2)

    def predict_proba(self, Xtest):
        # calculate x given y probabilities
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        normalizer = 1 / (
            2
            * np.power(np.pi, Xtest.shape[1] / 2)
            * np.sqrt(np.linalg.det(self.cov_matrix))
        )
        diff0 = Xtest - self.means0
        diff1 = Xtest - self.means1

        exp_term0 = -0.5 * np.einsum("ij,jk,ik->i", diff0, inv_cov_matrix, diff0)
        exp_term1 = -0.5 * np.einsum("ij,jk,ik->i", diff1, inv_cov_matrix, diff1)

        x_given_y0 = normalizer * np.exp(exp_term0)
        x_given_y1 = normalizer * np.exp(exp_term1)

        # calculate y given x probability for class 1
        denominator = x_given_y0 * self.prior_proba0 + x_given_y1 * self.prior_proba1
        proba = x_given_y1 * self.prior_proba1 / denominator

        return proba

    def get_params(self):
        params = super().get_params()
        params["cov_matrix"] = self.cov_matrix
        return params
