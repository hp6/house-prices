import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class MasVnrImputer(BaseEstimator, TransformerMixin):
    def __init__(self, type_fill_val="None", area_fill_val=0):
        self.type_fill_val = type_fill_val
        self.area_fill_val = area_fill_val
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.loc[X["MasVnrType"].isnull(), "MasVnrType"] = self.type_fill_val
        X.loc[X["MasVnrArea"].isnull(), "MasVnrArea"] = self.area_fill_val
        return X

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", columns=[]):
        self.strategy = strategy
        self.columns = columns
    
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy).fit(X.loc[:, self.columns])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.loc[:, self.columns] = self.imputer.transform(X.loc[:, self.columns])
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if len(self.columns) == 1:
            return X[self.columns]
        else:
            return X[self.columns]


    
