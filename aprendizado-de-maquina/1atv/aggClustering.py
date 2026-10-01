import matplotlib
matplotlib.use('Qt5Agg')  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Load preprocessed star dataset
df_processed = pd.read_csv('data/processed_data.csv')

# Separate feature matrix and target labels
numerical_columns = ['Temperature', 'L', 'R', 'A_M']
X = df_processed[numerical_columns]
y = df_processed['Type']

# Compute linkage matrix using Ward's minimum variance method -- could try other methods to see if balances the groups
linkage_matrix = linkage(X, method='ward')

if __name__ == '__main__':
    # --- Hierarchical Clustering Dendrogram ---
    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',  
        p=30,                  
        leaf_rotation=90.0,
        leaf_font_size=10.0
    )
    plt.axhline(y=12, color='r', linestyle='--', label='Cut threshold (k = 6)')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Groups / Cluster Merges')
    plt.ylabel('Euclidean Distance (Ward)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Model Fitting (k = 6 Clusters) ---
    k_ideal = 6
    hc = AgglomerativeClustering(
        n_clusters=k_ideal, 
        metric='euclidean', 
        linkage='ward'
    )
    df_processed['Cluster_HC'] = hc.fit_predict(X)

    print(f"Clustering complete. Data split into {k_ideal} clusters.\n")
    print("Sample distribution per cluster:")
    print(df_processed['Cluster_HC'].value_counts().sort_index())

    # --- Dimensionality Reduction for 3D Visualization ---
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # --- GRAPH 2: Interactive 3D Cluster Scatter Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=df_processed['Cluster_HC'],
        cmap='tab10',
        s=60,
        edgecolor='k',
        alpha=0.85
    )

    ax.set_title('3D Cluster Projection (PC1 vs PC2 vs PC3)', fontsize=14)
    ax.set_xlabel('Principal Component 1 (PC1)')
    ax.set_ylabel('Principal Component 2 (PC2)')
    ax.set_zlabel('Principal Component 3 (PC3)')

    # Add legend mapping colors to cluster IDs
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend)

    plt.tight_layout()
    plt.show()
