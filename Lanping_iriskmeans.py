import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plot = True

file = 'C:/Users/jessi/python_workspace/Iris.csv'
data = pd.read_csv(file)
data = data.drop(["Id"],axis=1) # remove ID column

if plot:
    sns.pairplot(data, hue = "Species")
    plt.show()
    
    sns.jointplot(data = data, x = "SepalLengthCm", y = "SepalWidthCm", hue = "Species", kind = "kde")
    plt.show()
    
    sns.countplot(x = "Species", data = data)
    plt.show()
  
    
labels = data.pop("Species")
# Clustering with 3 clusters
# a cross tab matrix to check how well has K-Means model classified the Species
kmeans = KMeans(n_clusters=3)
pred_labels = kmeans.fit_predict(data)
matrix = pd.DataFrame({'pred_species': pred_labels, 'species': labels})
ct = pd.crosstab(matrix['pred_species'], matrix['species'])
print(ct)
# Setoca has good result.

# Determing number of clusters
wss = []
silhouette = []
min_clusters = 2
max_clusters = 10
for k in range(min_clusters,max_clusters):
    kmeans = KMeans(n_clusters = k).fit(data)
    wss.append(kmeans.inertia_) # The elbow method
    silhouette.append(silhouette_score(data, kmeans.labels_)) # Silhouette score

# The elbow method
plt.plot(np.arange(min_clusters, max_clusters), wss)
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Within cluster sum of squares (WSS)")
plt.show()

# Silhouette scores
print(f"Silhouette scores: {silhouette}")
plt.plot(np.arange(min_clusters, max_clusters), silhouette)
plt.title("Silhouette Score")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show()

# From elbow method, k=2 or 3 is good. From Silhouette score k=2 is the best.
# Because there are 3 species, I prefer to choose k=3 and then Silehoutte score is the second largest.

