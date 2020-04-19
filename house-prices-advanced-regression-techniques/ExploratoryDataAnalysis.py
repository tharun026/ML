import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("train.csv")
#print(input_data.head(1))
#print(input_data.describe())

"""x = [1,2,3,4]
xarray = np.array(x)
print(x,xarray,xarray.shape)

xarray = xarray.reshape(1,4)
print(xarray,xarray.shape)

xarray = xarray.reshape(2,2)
print(xarray,xarray.shape)

x = [[1,2,4,8],[1,3,5,7]]
npx = np.array(x)
print(x)
print(npx,npx.shape)

npx = npx.reshape(8,1)
print(npx, npx.shape)"""

numeric_to_object = ['Id','MSSubClass','YrSold']
for col in numeric_to_object :
    df[col] = df[col].astype('object')

#print(input_data.describe())

#print(input_data['Id'].head(2))

five_columns = df.iloc[:,-5:]
#print(five_columns.head())
#print(input_data.columns)

null_columns = df.isna().any()
#print(null_columns)

na_list = [col_name \
           for col_name in df.columns \
               if null_columns[col_name]==True
           ]
#print(na_list)
#print(input_data.shape)
#print(na_list)

for col in na_list:
#    print(input_data[col])
    lotlist = np.array(df[col].fillna('x'))
#    print(lotlist)
    empty_x = len(lotlist[lotlist=='x'])
#    print(empty_x)
#    print(len(lotlist))
    naperc = empty_x/len(lotlist)
    
    print(col,";",naperc)
    if naperc>0.2:
        df = df.drop([col],axis=1)
print(df.shape)

df_num = df.select_dtypes(include = ['float64','int64'])
print(df_num.head())

df_num.hist(figsize=(20,20),bins=50)

#creating normalized values for each column
#getting the max value of each column and dividing each value by itself
#Scaling technique
#values will be changed to 0 - 1
df_norm = pd.DataFrame([])
for col in df_num.columns:
    df_norm[col] = df_num[col]/max(df_num[col])
print(df_norm.head())
