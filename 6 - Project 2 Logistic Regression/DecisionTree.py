import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split

df = pd.read_csv("German_Credit_data.csv")
#df.head()
#print(df.info())

numerical_columns = ['Duration_of_Credit_month','Credit_Amount','Percentage_of_disposable_income',
                    'Duration_in_Present_Residence','Age_in_years','No_of_Credits_at_this__Bank','No_of_dependents',
                    'Creditability']

categorical_columns = []

for i in df.columns:
    if i not in numerical_columns:
        df[i] = df[i].astype(str)
        categorical_columns.append(i)

for i in categorical_columns:
    dummies = pd.get_dummies(df[i],prefix=i)
    df = pd.concat([df,dummies],axis=1)

y = df['Creditability']
x = df.drop('Creditability',axis=1)
#print(x.info())
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0)

clf = dtc(random_state=0)
clf.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))


