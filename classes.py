import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

import functions as f


class ConflictingDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, inplace=False):
        self.inplace = inplace
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        X.drop(X.loc[(X["MasVnrType"] == "None") & (X["MasVnrArea"] != 0)].index, inplace=True)
        X.drop(X.loc[(X["MasVnrType"] == "BrkFace") & (X["MasVnrArea"] == 0)].index, inplace=True)
        X.drop(X.loc[(X["MasVnrType"] == "Stone") & (X["MasVnrArea"] == 0)].index, inplace=True)
        
        X.loc[(X["BsmtExposure"].isnull()) & ~(X["BsmtQual"].isnull()), "BsmtExposure"] = "No"


        return X

class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,columns=[], categories=[], sparse=False, handle_unknown="error"):
        self.columns = columns
        self.categories = categories
        self.sparse = sparse
        self.handle_unknown = handle_unknown
    
    def fit(self, X, y=None):
        if self.categories:
            self.one_hot_encoder = OneHotEncoder(categories=self.categories, sparse=self.sparse, handle_unknown=self.handle_unknown).fit(X[self.columns])
        else:
            self.categories = f.unique_values(X[self.columns])
            self.one_hot_encoder = OneHotEncoder(categories=self.categories, sparse=self.sparse, handle_unknown=self.handle_unknown).fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        ohe_matrix = self.one_hot_encoder.transform(X[self.columns])
        column_names = []
        for index, col_categories in enumerate(self.categories):
            for cat in col_categories:
                column_names.append("{}_{}".format(self.columns[index], str(cat))) 
        return  pd.concat([X, pd.DataFrame(ohe_matrix, columns=column_names)], axis=1)

class NumToCat(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], inplace=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        X[self.columns] = X[self.columns].astype(dtype="object")
        return X

class MasVnrImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        mas_vnr_types = f.unique_values(X[["MasVnrType"]])
        print(mas_vnr_types)
        for mas_vnr_type in mas_vnr_types:
            mean = X.loc[X["MasVnrType"] == mas_vnr_type, "MasVnrArea"].mean()
            X.loc[X["MasVnrArea"].isnull(), "MasVnrArea"] = mean
        return X

class NaCatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_val="NA",  columns=[]):
        self.fill_val = fill_val
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.columns] = X[self.columns].fillna(self.fill_val)
        # print(X.head())
        # print(X["PoolQC"])
        return X

class GarageYrImputer(BaseEstimator, TransformerMixin):
    def __init__(self, inplace=False):
        self.inplace = inplace
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()

        if ("GarageYrBlt" not in X.columns) or ("YearBuilt" not in X.columns):
            print("GarageYrBlt or YearBuilt missing")
            print(X.columns)
        else:
            X.loc[X["GarageYrBlt"].isnull(), "GarageYrBlt"] = X.loc[X["GarageYrBlt"].isnull(), "YearBuilt"]
        return X

class ConstantImputer(BaseEstimator, TransformerMixin):
    def __init__(self, string_fill_val="NA", number_fill_val=0, columns=[], inplace=False):
        self.string_fill_val = string_fill_val
        self.number_fill_val = number_fill_val
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        str_col = X[self.columns].select_dtypes(include=object).columns
        num_col = X[self.columns].select_dtypes(include="number").columns
        # print(str_col)
        for row in str_col:
            X[row].fillna(self.string_fill_val, inplace=True)
        for row in num_col:
            X[row].fillna(self.number_fill_val, inplace=True)
        return X

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", columns=[], inplace=False):
        self.strategy = strategy
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy).fit(X.loc[:, self.columns])
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()

        X.loc[:, self.columns] = self.imputer.transform(X.loc[:, self.columns])
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], inplace=False):
        self.columns = columns
        self.inplace = inplace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not self.inplace:
            X = X.copy()

        if len(self.columns) == 1:
            return X[self.columns]
        else:
            return X[self.columns]


    
