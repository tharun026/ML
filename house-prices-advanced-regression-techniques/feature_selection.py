import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier as ETC


df = pd.read_csv("train.csv")
#print(df.head())

col_list = ['Id','YrSold','MSSubClass','MoSold','BsmtFullBath','BsmtHalfBath','YearBuilt','OverallQual','YearRemodAdd']
for col in col_list:
    df[col] = df[col].astype('object')

df_num = df.select_dtypes(include = ['float64','int64'])
#print(df_num.head())

x = df_num.drop(['SalePrice'],axis=1)
cols = x.columns
z = np.nan_to_num(x)
y = df_num['SalePrice']
print(type(z))
print(type(y))

#print(x.head(),y.head())

estimator = SVR(kernel='linear')
selector = RFE(estimator,1,step=1)
selector = selector.fit(z,y)
rank = selector.ranking_

dic = {}
i = 0
for col in cols:
    dic[col] = rank[i]
    i = i+1

print(sorted(dic.items(),key=lambda kv: kv[1]))

imp = ETC(n_estimators=30)
imp = imp.fit(z,y)
importance = imp.feature_importances_

dic = {}
i = 0
for col in cols:
    dic[col] = importance[i]
    i = i+1

print(sorted(dic.items(),key=lambda kv: kv[1], reverse = True))


