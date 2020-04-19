import pandas as pd
import numpy as np
import os, sys
import re, string
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
import nltk
#nltk.download('stopwords')
nltk.download('wordnet')
from collections import Counter
from nltk.corpus import stopwords

df = pd.read_csv('labeledTrainData.csv',sep='\t',encoding='iso8859-1')
df=df.head(10000)
df_filtered = df[['sentiment','review']]
df_filtered = df_filtered.dropna(how='any')
df_filtered = df_filtered.drop_duplicates()

df1 = df_filtered[df_filtered['sentiment']==1]
df2 = df_filtered[df_filtered['sentiment']==0]
df_data = pd.concat([df1, df2])
print(df_data['sentiment'].value_counts())

def clean_tags(txt):
    txt=str(txt)
    txt=txt.strip();txt=txt.lower()
    txt=txt.replace('>','> ')
    txt=txt.replace('<',' <')
    txt=re.sub("[\<\[].*?[\>\]]", "", txt)
    return txt

def stopwords_list():
    stop_words_list = set(stopwords.words('english'))    
    stop_words_list.update(('and','a','so','arnt','this','when','It','many','so','cant','yes'))
    stop_words_list.update(('no','these','these','please', 'let', 'know', 'cant', 'can', 'pls', 'u', 'abt', 'wht'))
    return stop_words_list

stop_words_list = stopwords_list()    

def clean_text(text):  
    norm_text = text.lower()
    #remove use case specific keywords
    #norm_text = norm_text.replace('end report', ' ')         
    for char in ['\"', ',', '(', ')', '!', '?', ';', ':', '#', '*', '>','$']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
            
    norm_text = norm_text.replace('<U1177324', ' ')
    norm_text = norm_text.replace(' &nbsp',' ')
    norm_text = norm_text.replace('&nbsp',' ')
    norm_text = re.sub(r"\\", "", norm_text)    
    norm_text = re.sub(r"\'", "", norm_text)    
    norm_text = re.sub(r"\"", "", norm_text)  
    
    #clear number labels
    norm_text = re.sub('[0-9]{1,2}[.]', ' ', norm_text).strip() 
    #remove 1/20:
    norm_text = re.sub('[0-9]{1,2}[/][0-9]{1,2}[:]', ' ', norm_text).strip()
    #remove numbers
    norm_text = re.sub('[0-9]{1,2}[ ]', '', norm_text).strip()
    norm_text = re.sub('(\d{1,3}(?:\s*\d{3})*(?:,\d+)?)', ' ', norm_text).strip()

    #clear date
    norm_text = re.sub('[0-9]{1,2}[\/,:][0-9]{1,2}[\/,:][0-9]{2,4}', ' ', norm_text).strip() 
    return norm_text

#lemmatize and remove stop words
def lemmatize_text(text, stopwords_remove= True):
    lemmatizer = WordNetLemmatizer()
    if(stopwords_remove):
        resp = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words_list if word not in string.punctuation if word.isalpha()]
    else:#without stopword removal
        resp = [lemmatizer.lemmatize(word) for word in text.split() if word not in string.punctuation if word.isalpha()]
    return " ".join(resp)

# Master function to convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text, stopwords_remove = True):
    out1 = clean_tags(text); #print(1, out1)
    out2 = clean_text(out1); #print(2, out2)
    out3 = lemmatize_text(out2, stopwords_remove); #print(3, out3)
    return out3

# #test input
# normalize_text("<ff> dshfksj<> kajsflksuoiew sd325325")

for series in df_data.columns:
    if series == 'sentiment': continue
    df_data[series] = df_data[series].apply(normalize_text); print("completed--->", series)
    
#save processed data
df_data.to_csv("processed_data.csv", index=False)
#print(df_data.head())
