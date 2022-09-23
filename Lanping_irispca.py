# 1.importing libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# 2.reading the data
iris = load_iris() #iris is a dictionary, key-value informations
y = iris.target #get the target via key
X = iris.data #get eigenvector
print(X.shape) #as array with 2 dimensions
print(pd.DataFrame(X)) #as eigenvector with 4 dimensions

#3.standardizing the data
pca = PCA(n_components=2)#2 dimensions for visualisering
pca = pca.fit(X)
X_dr = pca.transform(X)#new eigenvector after dimension reduction
print(X_dr)

#4.visualisering
y #target, three viables showing three distributions of the data
#to display the data distribution of three irises in a two-dimensional plane coordinate system,
#the corresponding two coordinates(two features) are x1 and x2 of the three irises after dimension reduction
print(y)
X_dr[y== 0,0],X_dr[y==0,1]#boolean index

plt.figure()
plt.scatter(X_dr[y==0,0],X_dr[y==0,1],c="red",label=iris.target_names[0]) 
plt.scatter(X_dr[y==1,0],X_dr[y==1,1],c="black",label=iris.target_names[1])
plt.scatter(X_dr[y==2,0],X_dr[y==2,1],c="orange",label=iris.target_names[2])
plt.legend()
plt.title("PCA of IRIS dataset")
plt.show()
#visualisering, three clusters,showing the distribution of the three irises on the plane formed by the eigenvector

#5 value after dimension redecution
pca.explained_variance_ 
print(pca.explained_variance_)
#explained variance, the size of amount of the information carried by each new feature after dimension reduction
#the first feature has most of the information

pca.explained_variance_ratio_ 
print(pca.explained_variance_ratio_)
#explained variance ratio
#the percentage of information each new feature after dimension reduction occupies from the total original information
#the first feature has most of the information,92.46%

pca.explained_variance_ratio_.sum()
print(pca.explained_variance_ratio_.sum())
#sum of explained variance ratio, the new feature matrix retains 97.77% informaion
#reduce half dimension but retain 97.77% information
#nice dimension reduction result

#6 to se which number of componets is the best according to cumulative_explained_variance

pca_line = PCA().fit(X)# new eigenvector without dimensions reduction, 4 features
print(pca_line.transform(X))
print(pca_line.explained_variance_ratio_)# get ratio of information carried by every feature, most information are on the first feature
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))

plt.xticks([1,2,3,4])
#x axis displays 1234
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance")
plt.show()
#Four obvious points. If there are many features, chooset the point that makes the cumulative variance abruptly smooth.