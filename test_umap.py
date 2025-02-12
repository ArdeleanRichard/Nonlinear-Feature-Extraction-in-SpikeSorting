import umap
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
import warnings
warnings.filterwarnings("ignore")

from dataset_parsing import simulations_dataset as ds
simulation_number = 4
X, y_true = ds.get_dataset_simulation(simNr=simulation_number)
print(X.shape)

# umap: 10, 0.05, chebyshev:  0.8545120329353522 0.881592432435879 0.9440218451336064 0.15016106087175415 816.7560286649613 2.0671410135682615

scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

for n_neighbors in [2,3,5,10,15,20,30]:
    for min_dist in [0.01, 0.05, 0.1, 0.2]:
        for metric in ["euclidean", "chebyshev", "jaccard"]:
                model_umap = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_epochs=500)
                X_embedding = model_umap.fit_transform(X)

                model = KMeans(n_clusters=len(np.unique(y_true)))
                y_pred = model.fit_predict(X_embedding)

                ari = adjusted_rand_score(y_true, y_pred)
                ami = adjusted_mutual_info_score(y_true, y_pred)
                contingency_mat = contingency_matrix(y_true, y_pred)
                purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
                ss = silhouette_score(X, y_pred)
                chs = calinski_harabasz_score(X, y_pred)
                dbs = davies_bouldin_score(X, y_pred)

                print(f"{n_neighbors}, {min_dist}, {metric}: ", ari, ami, purity, ss, chs, dbs)
