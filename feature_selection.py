import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("train.csv")
print(df.head())

col_list = ['Id','YrSold','MSSubClass','MoSold','BsmtFullBath']
for col in col_list:
    df[col] = df[col].astype('object')

df_num = df.select_dtypes(include = ['float64','int64'])
print(df_num.head())
