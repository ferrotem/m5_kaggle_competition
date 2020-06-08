from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
import pandas as pd
matrix_root = "./matrixes"

scaled_train = np.load(matrix_root+"/scaled_train_values.npy")
scaled_price = np.load(matrix_root+"/scaled_price_values.npy")
diff_table = np.load(matrix_root+"/shifted.npy")
mean_table = np.load(matrix_root+"/total_mean.npy",allow_pickle=True)

def pca_calculation(x,i):
    pca = PCA(n_components=100, svd_solver='full')
    pca.fit(x[:,i:i+100].T)
    S_train = pca.transform(x[:,i:i+100].T)
    return S_train.T

X_train = []
for i in tqdm(range(1713)):
    X = []
    X.append(pca_calculation(scaled_train,i))
    X.append(pca_calculation(scaled_price,i))
    for el in diff_table:
        X.append(pca_calculation(el.T,i))
    for el2 in mean_table:
        X.append(pca_calculation(el2.T,i))
    X_train.append(X)
  
X_train = np.array(X_train)
np.save(matrix_root+"/pca_train_values.npy", X_train)


X_val = []
for i in tqdm(range(1713,1813)):
    X = []
    X.append(pca_calculation(scaled_train,i))
    X.append(pca_calculation(scaled_price,i))
    for el in diff_table:
        X.append(pca_calculation(el.T,i))
    for el2 in mean_table:
        X.append(pca_calculation(el2.T,i))
    X_val.append(X)

X_val = np.array(X_val)
np.save(matrix_root+"/pca_val_values.npy", X_val)