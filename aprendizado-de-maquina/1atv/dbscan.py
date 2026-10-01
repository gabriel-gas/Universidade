import matplotlib
matplotlib.use('Qt5Agg')  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv('data/processed_data.csv')
num_columns = ['Temperature', 'L', 'R', 'A_M']
X = df[num_columns]

nb = NearestNeighbors(n_neighbors=5)
nb.fit(X)

distances, _ = nb.kneighbors(X)

distances = np.sort(distances[:,-1])


plt.figure(figsize=(8, 4))
plt.plot(distances, '.' )
plt.title('K-distance Graph for DBSCAN eps tuning')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('5-NN Distance')
plt.grid(True)
plt.show()

dbscan = DBSCAN(eps=.6, min_samples=5)
clusters = dbscan.fit_predict(X)

df['DBSCAN_Cluster'] = clusters


n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_outliers = list(clusters).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of outliers (noise): {n_outliers}')


