from binary_classifier import BinaryClassifier
import numpy as np


class LDA(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.mu0 = None
        self.mu1 = None
        self.cov_matrix = None

    def fit(self, X, y):
        # calculate prior probabilities
        super().fit(X, y)

        # divide X by class
        X0 = X[y == 0]
        X1 = X[y == 1]
        n0 = len(X0)
        n1 = len(X1)

        # calculate mean for class 0 and class 1
        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)

        # calculate covariance matrix
        cov0 = np.cov(X0, rowvar=False)
        cov1 = np.cov(X1, rowvar=False)
        self.cov_matrix = (cov0 * (n0 - 1) + cov1 * (n1 - 1)) / (n0 + n1 - 2)

    def predict_proba(self, Xtest):
        # calculate x given y probabilities
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        normalizer = 1 / (
            2
            * np.power(np.pi, Xtest.shape[1] / 2)
            * np.sqrt(np.linalg.det(self.cov_matrix))
        )
        diff0 = Xtest - self.mu0
        diff1 = Xtest - self.mu1

        exp_term0 = -0.5 * np.einsum('ij,jk,ik->i', diff0, inv_cov_matrix, diff0)
        exp_term1 = -0.5 * np.einsum('ij,jk,ik->i', diff1, inv_cov_matrix, diff1)

        x_given_y0 = normalizer * np.exp(exp_term0)
        x_given_y1 = normalizer * np.exp(exp_term1)

        # calculate y given x probability for class 1
        denominator = x_given_y0 * self.prior_proba0 + x_given_y1 * self.prior_proba1
        proba = x_given_y1 * self.prior_proba1 / denominator

        return proba

    def predict(self, Xtest, treshold=0.5):
        proba = self.predict_proba(Xtest)
        return (proba >= treshold).astype(int)

    def get_params(self):
        prior_proba0, prior_proba1 = super().get_params()
        return prior_proba0, prior_proba1, self.mu0, self.mu1, self.cov_matrix


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from dataset_generation import generate_dataset

    Xtrain, ytrain = generate_dataset(1000, 2, (0, 10), (1,1), 123)
    Xtest, ytest = generate_dataset(1000, 2, (0, 10), (1, 1), 321)

    lda_classifier = LDA()

    lda_classifier.fit(Xtrain, ytrain)
    ypred = lda_classifier.predict(Xtest)

    print(accuracy_score(ytest, ypred))