import pandas as pd
import numpy as np

def lists_eq(list1, list2):
    return np.sum(np.sort(list1) == np.sort(list2)) == len(list1)

def unique_values(df):
    unique_values = []
    for column in df:
        unique_values.append(df[column].unique())
    return unique_values