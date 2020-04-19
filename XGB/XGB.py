import pandas as pd
import numpy as np
import os, sys
import re, string
import seaborn as sns


df = pd.read_csv('train.csv')
print(df.shape)
print(df.info())
print(df.describe(include='all'))

y = df['SalePrice']
print(y.head())

print(sns.distplot(y,hist=True))

