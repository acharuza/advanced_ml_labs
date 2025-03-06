from .binary_classifier import BinaryClassifier
import numpy as np


class QDA(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.cov_matrix0 = None
        self.cov_matrix1 = None

    def fit(self, X, y):
        # calculate prior probabilities and means
        super().fit(X, y)

        # calculate coviarance matrices
        self.cov_matrix0 = np.cov(self.X0, rowvar=False)
        self.cov_matrix1 = np.cov(self.X1, rowvar=False)

        # adding small value to the diagonal to ensure matrix is invertible
        self.cov_matrix0 += np.eye(self.cov_matrix0.shape[0]) * 1e-6
        self.cov_matrix1 += np.eye(self.cov_matrix1.shape[0]) * 1e-6

    def predict_proba(self, Xtest):
        # calculate x given y probabilities
        inv_cov_matrix0 = np.linalg.inv(self.cov_matrix0)
        inv_cov_matrix1 = np.linalg.inv(self.cov_matrix1)
        normalizer0 = 1 / (
            2
            * np.power(np.pi, Xtest.shape[1] / 2)
            * np.sqrt(np.linalg.det(self.cov_matrix0))
        )
        normalizer1 = 1 / (
            2
            * np.power(np.pi, Xtest.shape[1] / 2)
            * np.sqrt(np.linalg.det(self.cov_matrix1))
        )
        diff0 = Xtest - self.means0
        diff1 = Xtest - self.means1

        exp_term0 = -0.5 * np.einsum("ij,jk,ik->i", diff0, inv_cov_matrix0, diff0)
        exp_term1 = -0.5 * np.einsum("ij,jk,ik->i", diff1, inv_cov_matrix1, diff1)

        x_given_y0 = normalizer0 * np.exp(exp_term0)
        x_given_y1 = normalizer1 * np.exp(exp_term1)

        # calculate y given x probability for class 1
        denominator = x_given_y0 * self.prior_proba0 + x_given_y1 * self.prior_proba1
        proba = x_given_y1 * self.prior_proba1 / denominator

        return proba

    def get_params(self):
        params = super().get_params()
        params["cov_matrix0"] = self.cov_matrix0
        params["cov_matrix1"] = self.cov_matrix1
        return params
