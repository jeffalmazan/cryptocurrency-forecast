import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# from read_csv import read_data
# from preprocessing import perform_cleanup
# df = read_data()
# df = perform_cleanup(df)

def apply_pca(df, n_components=None):
    
    # Extracting the relevant features for PCA
    features = ['Open', 'High', 'Low']
    X = df[features]
    
    # Standardizing the features before applying PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sets variance automatically to 95% if n_components is set to none
    if n_components is None:
        pca = PCA(n_components=0.95)
    else:
        pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Printing the components
    print("PCA components:\n", pca.components_)
    
    # Creating a DataFrame with the PCA features
    pca_columns = [f'PCA_{i+1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
    
    # Combining the PCA features with the original DataFrame
    df_with_pca = pd.concat([df, df_pca], axis=1)
    
    # Printing the new features
    print("New PCA features:\n", df_with_pca.head())
    
    return df_with_pca

# df_with_pca = apply_pca(df)

