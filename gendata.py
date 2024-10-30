import numpy as np
from numpy.random import default_rng
rng = default_rng()

# Generate noise
def gen_noise(n, sigma = None):
    if sigma is None:
        sigma = np.identity(n)
    mean = np.zeros(n)
    return np.random.multivariate_normal(mean = mean, cov = sigma)
# Generate data 
def gen(n, p, num_outliers, delta, sigma = None):
    # Generate X  
    X = []
    for i in range(n):
        Xi = []
        for _ in range(p):
            Xi.append(np.random.normal(0, 1))  
        X.append(Xi)
    # Set Beta
    Beta = np.array([((i%2) + 1) for i in range(p)]) 
    # Generate y
    y = []
    for i in range(n):
        mu = np.dot(X[i], Beta)
        y.append(mu)
    noise = gen_noise(n, sigma)
    y = y + noise
    
    if num_outliers != 0:
        # Randomly select "num_outliers" outliers
        ids = rng.permutation(n)
        IsOutlier = ids[: num_outliers]
        # Adding a shift
        for i in IsOutlier:
            rand = np.random.rand()
            sign = 1
            if rand < 0.5:
                sign = -1
            y[i] += sign*delta

    y = np.array(y).reshape((n, 1))
    X = np.array(X)
    if num_outliers == 0:
        return X, y
    return X, y, sorted(IsOutlier)