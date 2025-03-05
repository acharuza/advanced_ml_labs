import numpy as np

N_SAMPLES = 1000
N_FEATURES = 2


def generate_dataset1(a):
    y = np.random.binomial(1, 0.5, N_SAMPLES)
    X = np.where(
        y[:, None] == 0,
        np.random.normal(loc=0, scale=1, size=(N_SAMPLES, N_FEATURES)),
        np.random.normal(loc=a, scale=1, size=(N_SAMPLES, N_FEATURES)),
    )
    return X, y


def generate_dataset2(a, ro):
    y = np.random.binomial(1, 0.5, N_SAMPLES)
    cov_matrix0 = np.array([[1, ro], [ro, 1]])
    cov_matrix1 = np.array([[1, -ro], [-ro, 1]])
    X = np.where(
        y[:, None] == 0,
        np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix0, size=N_SAMPLES),
        np.random.multivariate_normal(mean=[a, a], cov=cov_matrix1, size=N_SAMPLES),
    )
    return X, y
