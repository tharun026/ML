import numpy as np # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import pickle
import operator
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

#pre process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

# algos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# model selection
from sklearn import model_selection
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# analysis
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

url = 'H:\Imarticus\All_Classification\imdb-sentiments\\train.csv'
features = ['text','sentiment']

def cleanText(text, removeStopwords=True, performStemming=True):
    
    #regex for removing non-alphanumeric characters and spaces
    remove_special_char = re.compile('r[^a-z\d]', re.IGNORECASE)
    #regex to replace all numerics
    replace_numerics = re.compile(r'\d+', re.IGNORECASE)
    text = remove_special_char.sub('', text)
    text = replace_numerics.sub('', text)

    stop_words = set(stopwords.words('english')) 
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    #convert text to lowercase.
    text = text.lower().split()

    
    processedText = list()
    for word in text:        
        if removeStopwords:
            if word in stop_words:
                continue
        if performStemming:
            word = stemmer.stem(word)
            
        word = lemmatizer.lemmatize(word)
        word = lemmatizer.lemmatize(word, 'v')
            
        processedText.append(word)

    text = ' '.join(processedText)

    return text

def vectorize(data):
    if type(data)==str: data=data.split()
    cdata=set(data)
    vector_map={}; i=0
    for c in cdata:
        vector_map[i]=c; i=i+1
    #print(vector_map)
    return vector_map


def sort_by_value(dictx):
    dicty=sorted(dictx.items(),key=operator.itemgetter(1),reverse=True)
    return dicty
    

def tex2vec(X):
    xlist=[]
    if type(X)==str: X=[X]; print("txt2vec--->string")
    if type(X)==list: xlist=X; print("txt2vec--->list")
    if type(X)==np.ndarray:
        for i in range(len(X)): xlist.append(X[i][0])
        print("txt2vec--->ndarray"); #print(X[i][0]); return
        
    vectorizer = TfidfVectorizer(min_df=0.001, max_df=1.0)
    trainvectors = vectorizer.fit_transform(xlist)
    return trainvectors

#tv = tex2vec(["i am on the train and travelling to home town", "my home town is traditionally rich"]); print(tv) 

def get_XY(url, vectorize, features):
    dataset = pandas.read_csv(url, names=features, encoding="ISO-8859-1"); #print(dataset)
    if vectorize==1:
        h=list(dataset.columns.values)[0]; #print(h)
        dataset[h] = dataset[h].values.astype('U')
    array = dataset.values; #print(array[0])
    n = len(array[0]); #print("len--->", n)
    X = array[:,0:n-1]
    Y = array[:,n-1]
    return X, Y

#X, Y = get_XY(url, 0, features); print(X[:5], Y[:5])



def pca_prog(url, features, n):
    X, Y =  get_XY(url, 0, features)
    X, Y = X[1:], Y[1:]
    pca = PCA(n_components=n)
    fit = pca.fit(X);
    print(fit.explained_variance_ratio_)
#    print(fit.singular_values_)
    xfit = fit.components_; #print(xfit[:5])
    return xfit

#pca_prog(url, features, 4)


def rfe_prog(url, features, n):
    X, Y =  get_XY(url, 0, features)
    X, Y = X[1:], Y[1:]
    model = LogisticRegression()
    rfe = RFE(model, n)
    fit = rfe.fit(X,Y)
    xfit = fit.ranking_; print(xfit)
    return xfit

#rfe_prog(url, features, 1)

def ETC_prog(url, features, n):
    X, Y =  get_XY(url, 0, features)
    X, Y = X[1:], Y[1:]
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    f_imp = model.feature_importances_; print(f_imp)
    print(X[:2])
    return f_imp

#ETC_prog(url, features, 4)

def top_fits(X, n, f_imp):
    importance={}
    for i in range(len(f_imp)):
        importance[i]=round(f_imp[i],3)
    importance = sort_by_value(importance); print(importance)
    
    XT = X.transpose(); xNew=[]; c=0
    for k, v in importance:
        xNew.append(XT[k])
        if c>=n: break
        c=c+1
    xNew = np.array(xNew)
    X = xNew.transpose()
    return X

#top_fits(X, n, f_imp)
def summarize(url, features):
    dataset = pandas.read_csv(url, names=features)
    newdata = pandas.DataFrame([])
    for col in dataset.columns:
        if col=="class": continue
        newdata[col]=np.log(dataset[col])

    #Summarize the Dataset
    Summary={}
    Summary['Shape']=dataset.shape; print(Summary['Shape'])
    Summary['Describe']=dataset.describe(); print(Summary['Describe'])
    Summary['Groups']=dataset.groupby('class').size().to_json(); print(Summary['Groups'])
      
#    Data Visualization
#    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False); plt.savefig("box.jpg"); plt.show()
#    dataset.hist(); plt.savefig("hist.jpg"); plt.show()
#    newdata.hist(); plt.show()
#    scatter_matrix(dataset); plt.savefig("scatter.jpg"); plt.show()

#summarize(url, features) 
 
def get_models():
    models = {}
    models['LogR'] = LogisticRegression()
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier()
    models['DTC'] = DecisionTreeClassifier(criterion="gini")
    models['NBC'] = GaussianNB()
    models['SVC'] = SVC()
    models['RFC'] = RandomForestClassifier()
    models['MLP'] = MLPClassifier()
    models['GBC'] = GradientBoostingClassifier()
    return models

##models = get_models();
##for k, v in models.items(): print(k, ":", v); print("")   

def compare(url, features, vectorize):
    X, Y = get_XY(url, vectorize, features) 
    if vectorize==1: X = tex2vec(X); X = X.toarray(); #print(X)
    
    #create validation set
    validation_size = 0.20
    seed = 7
    Xt, Xv, Yt, Yv = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    #Test options and evaluation metric
    scoring = 'accuracy'

    models = get_models()
    # evaluate each model in turn
    results = []
    model_names = []
    model_list = {}
    compare_list = {}
    for name, model in models.items():
        kfold = model_selection.KFold(n_splits=2, random_state=seed)
        cv_results = model_selection.cross_val_score(model, Xt, Yt, cv=kfold, scoring=scoring)
        results.append(cv_results)
        model_names.append(name)
        model_list[model]=cv_results.mean()
        compare_list[name]=[" Mean: "+str(round(cv_results.mean(),2)), "  Std: "+str(round(cv_results.std(),2))]
        print(name, ':', cv_results.mean(), cv_results.std())
    #print(model_names)
    model_dict=sort_by_value(model_list); print(model_dict)
    final_model=model_dict[0][0]; print('final_model: ',final_model)

    #Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(model_names)
    plt.savefig('comparison.jpg'); plt.show()
    print(compare_list)
    #return [compare_list, "C:/services.ai/Classifier/comparison.jpg", str(final_model)]


#compare(url, features, 0)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(); print(title); print(y)
    plt.title(title)
    if ylim is not None: plt.ylim(*ylim); print(1)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=2)
    print(2)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print(3)
    plt.grid()
    print(4)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    print(5)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    print(6)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    print(7)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    print(8)
    plt.legend(loc="best"); plt.show()
    print("end of learning curve")
	#plt.savefig("learning_curve.jpg"); plt.show()
    return plt


    

def train(url, features, model_key, vectorize, model_name):
    model_dict=get_models()
    final_model=model_dict[model_key]; #print('final_model--->', final_model)
    X, Y = get_XY(url, vectorize, features); #print("input-->", X[0]); print("label-->", Y[0])
    if vectorize==1: X = tex2vec(X); X = X.toarray(); print(X.shape)
    
    # selecting importanct features
    #n = 10
    #X = pca_prog(url, features, n)
    #X = ETC_prog(url, features, n)
    
    #create validation set
    validation_size = 0.20
    seed = 7
    Xt, Xv, Yt, Yv = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

##    fpr, tpr, th = roc_curve(Yt, Yv)
##    roc_auc = auc(fpr, tpr); print('roc_auc-->', roc_auc)
    
    ## Make predictions on validation dataset
    Yt = Yt.reshape(Yt.size, 1)
    final_model.fit(Xt, Yt)
    predictions = final_model.predict(Xv)
    score = accuracy_score(Yv, predictions)
    report = classification_report(Yv, predictions)
    matrix = confusion_matrix(Yv, predictions)
    print('Accuracy: ', score); 
    print(""); print(report); print(""); print(matrix)
    pickle.dump(final_model, open('models/'+model_name+'.pickle','wb')); print("Training Completed")
    
    title = "Learning Curves - "+str(model_key)
    # Cross validation with 2 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
#    Y = Y.reshape(150,1)
    y=Y
    plot_learning_curve(final_model, title, X, y)
    plt.show()

#review = pandas.read_csv(url)
#print(review.head(5))
#review['sentiment'].value_counts()

#raw_data = review['text'].tolist()

#pre_vector = [cleanText(review) for review in raw_data]

model_key = "LogR"
vectorize = 0
model_name = model_key
train(url, features, model_key, vectorize, model_name) 





