import numpy as np
from numpy.random import default_rng
rng = default_rng()
from numpy.linalg import pinv

# Linear Regression using LS method
class LinearRegressionModel:
    def __init__(self):
        super().__init__()
        self.beta = None
    def fit(self, X, y):
        self.beta = np.dot(pinv(X), y)
    def predict(self, X):
        return np.dot(X, self.beta)
# RANSAC
class RANSAC:
    def __init__(self, B = None, t = None):
        self.B = B                                  # Number of iterations
        self.t = t                                  # Threshold value to determine if points are fit well
        # If B and t were set to None, we will set t = 5 and calculate B by default
        self.largest = 0                            # The largest number of inliers
        self.opt_model = LinearRegressionModel()    # The optimal model
        self.best_outliers = []                     # Outlier set of the optimal model
        self.maybe_inliers_set = []                 # Set of maybe inliers, the i'th element is a set that consists maybe inliers selected in the i'th iteration

    def fit(self, X, y):
        n = X.shape[0]
        p = X.shape[1]
        if self.B is None:
            self.B = int(np.log(0.01)/np.log(1 - (0.8**(p + 1))))
        if self.t is None:
            self.t = 5
        
        for i in range(self.B):
            # Get p + 1 random data points from dataset to be maybe inliers
            ids = rng.permutation(n)
            maybe_inliers = ids[: (p + 1)]
            self.maybe_inliers_set.append(maybe_inliers)
            # Fit model with selected maybe inliers
            maybe_model = LinearRegressionModel()
            maybe_model.fit(X[maybe_inliers], y[maybe_inliers])
            # Classify data points as inlier or outlier
            outliers = []
            inliers = []
            for j in range(n):
                if self.SE(y[j], maybe_model.predict(X[j])) > self.t:
                    outliers.append(j)
                else:
                    inliers.append(j)
            # Saving the optimal model
            if len(inliers) > self.largest:
                self.largest = len(inliers)
                self.best_outliers = outliers
                self.opt_model.fit(X[inliers], y[inliers])
        return self
    
    def predict(self, X):
        return self.opt_model.predict(X)
    
    def SE(self, y_true, y_pred):
        return (y_true - y_pred)**2