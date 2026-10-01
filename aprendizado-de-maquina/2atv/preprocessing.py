import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib 
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath = 'superconductivty+data/train.csv'):

    df = pd.read_csv(filepath)

    X = df.drop(columns = 'critical_temp')
    Y = df['critical_temp']

    return X, Y

def get_train_test_split(X, Y, test_size = .2, random_state = 42):
    return train_test_split(X,Y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_train_scaled, scaler

if __name__ == '__main__':
    print('Exploratory Analysis... \n')
    X, Y = load_data()

    print(f'Target variable distribution')
    plt.figure(figsize=(8,5))
    sns.histplot(Y, bins = 50, kde = True, color = 'purple')
    plt.title('Critical Temperature Distribution')
    plt.xlabel('Critical Temperature (K)')
    plt.ylabel('Frequency')
    plt.grid(linestyle = ':')
    plt.show()


    print(f'Dataset shape: X = {X.shape}, Y = {Y.shape}')

    missing_values = X.isnull().sum().sum() + Y.isnull().sum()
    print(f'Total missing values: \n {missing_values}\n')

    Q1, Q3 = Y.quantile(.25), Y.quantile(.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5*IQR
    outliers_count = (Y > upper_bound).sum()
    print(f'Outliers detected in critical_temp above the upper bound {upper_bound:.2f}(K): {outliers_count} \n')

    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    high_corr_count = sum(any(upper_tri[col] > .95) for col in upper_tri.columns)
    print(f'high correlation count: {high_corr_count}\n')