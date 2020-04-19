import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

df = pd.read_csv('labeledTrainData.csv',sep='\t')
reviews = df.review.tolist()
print(len(reviews))

vec = TfidfVectorizer(stop_words = 'english',max_features=20000)
print('vec')
print(vec)
vec_data = vec.fit_transform(reviews)
print('vec_data')
print(vec_data)
vec_norm = normalize(vec_data)
print('vec_norm')
print(vec_norm)
vec_array = vec_norm.toarray()
print('vec_array')
print(vec_array)
vec_pca = PCA(n_components = 2).fit_transform(vec_array)
print('vec_pca')
print(vec_pca)

kmeans = KMeans(n_clusters=2, max_iter = 100, algorithm = 'auto')
fitted = kmeans.fit(vec_pca)
predictions = kmeans.predict(vec_pca)

plt.scatter(vec_pca[:,0],vec_pca[:,1],c=predictions,s=50)
#number_of_clusters = range(1,4)
#
#kmeans = [KMeans(n_clusters=i, max_iter=100) for i in number_of_clusters]
#
#score = [kmeans[i].fit(vec_array) for i in range(len(kmeans))]
#
#plt.plot(number_of_clusters,score); plt.show()