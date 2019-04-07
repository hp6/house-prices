import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MasVnrImputer(BaseEstimator, TransformerMixin):
    def __init__(self, type_fill_val="None", area_fill_val=0):
        self.type_fill_val = type_fill_val
        self.area_fill_val = area_fill_val
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[X["MasVnrType"].isnull(), "MasVnrType"] = self.type_fill_val
        X.loc[X["MasVnrArea"].isnull(), "MasVnrArea"] = self.area_fill_val
        return X
        
    
