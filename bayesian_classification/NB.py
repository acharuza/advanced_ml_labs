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

        def gaussian_density(x, mean, variance):
            first_term = 1 / np.sqrt(2 * np.pi * np.square(variance))
            second_term = np.exp(-np.square(x - mean) / (2 * variance))
            return first_term * second_term

        # calculate x given y probabilities
        x_given_y0 = np.prod(gaussian_density(Xtest, self.means0, self.variances0), axis=1)
        x_given_y1 = np.prod(gaussian_density(Xtest, self.means1, self.variances1), axis=1)

        # calculate y given x probability for class 1
        denominator = x_given_y0 * self.prior_proba0 + x_given_y1 * self.prior_proba1
        proba = x_given_y1 * self.prior_proba1 / denominator

        return proba

    def get_params(self):
        params = super().get_params()
        params["variances0"] = self.variances0
        params["variances1"] = self.variances1
        return params


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from dataset_generation import generate_dataset

    Xtrain, ytrain = generate_dataset(1000, 2, (0, 10), (1, 2), 123)
    Xtest, ytest = generate_dataset(1000, 2, (0, 10), (1, 2), 321)

    nb_classifier = NB()

    nb_classifier.fit(Xtrain, ytrain)
    ypred = nb_classifier.predict(Xtest)
    print(ypred)
    print(accuracy_score(ytest, ypred))
