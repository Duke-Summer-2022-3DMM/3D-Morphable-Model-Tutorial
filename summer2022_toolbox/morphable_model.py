import numpy as np

# define model class
class MorphableModel:
    def __init__(self, X_avg, EigenVec, n_features):
        self.X_avg = X_avg
        self.V = EigenVec
        self.n_features = n_features

    def fit(self, test_shape):
        features = self.V[:, :self.n_features]
        weights = (test_shape - self.X_avg.flatten('F')) @ features
        fitted_model = self.X_avg.flatten('F') + features @ weights
        return fitted_model