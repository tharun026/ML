import pandas as pd
import numpy as np

from sklearn.externals import joblib
import _pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC
import os

#pull the processed data
df_data = pd.read_csv("processed_data.csv")

#dropna, if missed in preprocess
df_data = df_data.dropna(how='any')

print(df_data.head(1))

'''
i=0
for s in df_data.columns:
    if i==0:
        df_data["combined"] = df_data[s]
    else:
        df_data["combined"] = df_data["combined"] + df_data[s]
    i=i+1
    
#print(df_data.head(1))
'''
x = df_data["review"]
y = df_data['sentiment']
# print(x.head(1))

#vectorize
vectorizer = CountVectorizer()
vec_train = vectorizer.fit_transform(x)
feature_file = "pfwt_vectors.pkl"
pickle.dump(vectorizer.vocabulary_, open(feature_file,"wb"))

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_file, "rb")))
transformer = TfidfTransformer()
x_vec = transformer.fit_transform(loaded_vec.fit_transform(x))
print(x_vec.shape)

import warnings
warnings.filterwarnings('ignore')

x_vec = x_vec.toarray()
# #split train and test
xtrain, xtest, ytrain, ytest = train_test_split(x_vec, y)
print(len(xtrain), len(xtest)); print(len(ytrain), len(ytest))

for n in [30]:
    print("Results for estimator size : "+str(n)); print("")
    classifier = RFC(bootstrap=True, class_weight=None, criterion='gini',
     max_depth=None, max_features='auto', max_leaf_nodes=None,
     min_impurity_split=1e-07, min_samples_leaf=2,
     min_samples_split=2, min_weight_fraction_leaf=0.0,
     n_estimators=n, n_jobs=1, oob_score=False, random_state=None,
     verbose=0, warm_start=False)

    classifier.fit(xtrain, ytrain)
    
    train_predictions = classifier.predict(xtrain)
    train_accuracy = accuracy_score(train_predictions, ytrain); print("training accuracy: ", train_accuracy); print("")
    print(classification_report(ytrain, train_predictions)); print("")

    test_predictions = classifier.predict(xtest)
    test_accuracy = accuracy_score(test_predictions, ytest); print("testing accuracy: ", test_accuracy); print("")
    print(classification_report(ytest, test_predictions)); print("")
    
#save model
model_file = 'pfwt_model.sav'
joblib.dump(classifier, model_file)
