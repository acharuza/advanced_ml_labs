import numpy as np

def generate_dataset(n_samples, n_features, mean, variance, seed):
    np.random.seed(seed)
    y = np.random.binomial(1, 0.5, n_samples)
    n1 = np.sum(y)
    n0 = len(y) - n1
    X = np.where(y[:, None] == 0,
                 np.random.normal(mean[0], variance[0], (n_samples, n_features)),
                 np.random.normal(mean[1], variance[1], (n_samples, n_features)))
    return X, y    