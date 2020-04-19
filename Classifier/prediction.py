
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
import _pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.utils import resample

#dev_loc = r'/axp/buanalytics/csswcpfwt/dev/'
dev_loc=os.path.dirname(os.path.abspath(__file__))+'/'


# In[4]:


#pull the processed data
df_data = pd.read_csv(dev_loc+"processed_data.csv")

#dropna, if missed in preprocess
df_data = df_data.dropna(how='any')

# for testing I have sampled, but in real-time it's not needed.
#df_data = resample(df_data, replace=False, n_samples=1000, random_state=123)

df_data.head(1)


# In[10]:


# get x, y

i=0
for s in df_data.columns:
    if i==0:
        df_data["combined"] = df_data[s]
    else:
        df_data["combined"] = df_data["combined"] + df_data[s]
    i=i+1
    
#print(df_data.head(1))

x = df_data["combined"]
y = df_data['feature_pfwt']
print(x.head(1))


# In[14]:



filename = dev_loc+'pfwt_model.sav'
feature_file = dev_loc+"pfwt_vectors.pkl"

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_file, "rb")))
transformer = TfidfTransformer()
x_vec = transformer.fit_transform(loaded_vec.fit_transform(x))

loaded_model = joblib.load(filename)
test_predictions = loaded_model.predict(x_vec)
test_accuracy = accuracy_score(test_predictions, y); print("testing accuracy: ", test_accuracy); print("")
print(classification_report(y, test_predictions)); print("")

