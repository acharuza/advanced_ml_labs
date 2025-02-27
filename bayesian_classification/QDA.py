from bayesian_classification.binary_classifier import BinaryClassifier

class QDA(BinaryClassifier):

    def __init__(self):
        super().__init__()
        self.mu0 = None
        self.mu1 = None
        self.cov_matrix0 = None
        self.cov_matrix1 = None