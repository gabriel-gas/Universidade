import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load preprocessed star dataset
df_processed = pd.read_csv('data/processed_data.csv')

# Separate predictor variables and ground truth target
numerical_columns = ['Temperature', 'L', 'R', 'A_M']
X = df_processed[numerical_columns]
y = df_processed['Type']

# Range of cluster numbers to evaluate
k_range = range(2, 11)
inertias = []
silhouette_scores = []

# Calculate inertia and silhouette score for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

if __name__ == '__main__':
    # --- Elbow Method & Silhouette Score Evaluation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow curve (Inertia vs k)
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Elbow Method (Inertia vs k)')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (WCSS)')
    ax1.grid()
    ax1.legend()

    # Silhouette score curve (Silhouette vs k)
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Silhouette Score vs k')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()