import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='IQR', factor=1.5):
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        if self.method == 'IQR':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - self.factor * IQR
            self.upper_bound = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        X_clipped = X.clip(lower=self.lower_bound, upper=self.upper_bound, axis=1)
        return X_clipped