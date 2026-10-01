import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''Pré-Processing'''

df_stars = pd.read_csv('data/Stars.csv')
df_categories = pd.read_csv('data/Categorias.csv')

df_stars = pd.merge(df_stars, df_categories, on='Star')


#Strings
df_stars['Color'] = df_stars['Color'].str.lower().str.strip()

color_mapping = {
    'blue white': 'blue-white',
    'blue-white': 'blue-white',
    'yellowish white': 'yellow-white',
    'white-yellow': 'yellow-white',
    'yellowish': 'yellow',
    'whitish': 'white',
    'pale yellow orange': 'orange-yellow'

}

df_stars['Color'] = df_stars['Color'].replace(color_mapping)


#Numerical Values
df_stars = df_stars[df_stars['A_M'].abs() <= 50]
log_columns = ['Temperature', 'L', 'R']
df_stars[log_columns] = np.log10(df_stars[log_columns])

df_processed = pd.get_dummies(df_stars, columns=['Color', 'Spectral_Class'])
scaler = StandardScaler()
numerical_columns = ['Temperature', 'L', 'R', 'A_M']
df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])

#Saving
df_processed.to_csv('data/processed_data.csv', index=False)


#Graphs
if __name__ == '__main__':

    #Single PCA fit on predictor variables
    X = df_processed[numerical_columns]
    y = df_processed['Type']

    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Number of components needed for 90% variance
    c_var = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(c_var >= 0.90) + 1
    print(f'Number of components required for >= 90% variance: {num_components}')

    # Cumulative Explained Variance 
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(c_var) + 1), c_var, marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
    plt.axvline(x=num_components, color='g', linestyle='--', label=f'{num_components} Components')
    plt.title('Cumulative Explained Variance by PCA')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Proportion of Explained Variance')
    plt.legend()
    plt.grid(alpha = .6)
    plt.tight_layout()
    plt.show()

    #Weights
    num_pcs = len(X.columns)
    pc_names = [f'PC{i+1}' for i in range(num_pcs)]
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=pc_names,
        index=X.columns
    )

    for pc in pc_names:
        print(f"\n--- Top 3 most important variables in {pc} ---")
        top_vars = loadings[pc].abs().sort_values(ascending=False).head(3)
        print(top_vars)



