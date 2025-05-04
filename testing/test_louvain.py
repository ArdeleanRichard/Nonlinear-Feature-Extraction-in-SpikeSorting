# Louvain: Available in the community module.
import community as community_louvain
import networkx as nx
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.sparse import csgraph
from sknetwork.topology import get_largest_connected_component
from dataset_parsing import simulations_dataset as ds


simulation_number = 4
X, y_true = ds.get_dataset_simulation(simNr=simulation_number)

transformer = PCA(n_components=2)
X_transformed = transformer.fit_transform(X)


# Step 1: Build a graph using k-nearest neighbors (k-NN)
knn = NearestNeighbors(n_neighbors=50)  # You can adjust k here
knn.fit(X_transformed)
distances, indices = knn.kneighbors(X_transformed)

# Create a graph
G = nx.Graph()

# Step 2: Add nodes and edges to the graph
for i in range(len(X_transformed)):
    G.add_node(i, pos=X_transformed[i])

# Add edges based on k-NN
for i in range(len(X_transformed)):
    for j in indices[i][1:]:  # Skip the self-loop (the nearest neighbor is the point itself)
        G.add_edge(i, j)

partition = community_louvain.best_partition(G)
print(partition)  # Dictionary of node assignments


labels = np.array([partition[node] for node in G.nodes])
print(labels)


plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title("Louvain Clustering on Iris Dataset (Graph-based)")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()







# # SCIKIT VERSION DOES NOT WORK - TypeError: No matching signature found
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy import sparse
# from scipy.sparse import csr_matrix
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
# from sklearn.neighbors import kneighbors_graph
# from sknetwork.clustering import Louvain
# from scipy.sparse import csgraph
# from sknetwork.topology import get_largest_connected_component
# from dataset_parsing import simulations_dataset as ds
#
#
# from sknetwork.data import karate_club
#
# adjacency = karate_club()
#
# # simulation_number = 4
# # X, y_true = ds.get_dataset_simulation(simNr=simulation_number)
# #
# # transformer = PCA(n_components=2)
# # X_transformed = transformer.fit_transform(X)
# #
# # adjacency = pairwise_distances(X_transformed)
# # adjacency = adjacency>0.5
# # adjacency = sparse.csr_matrix(adjacency)
# # print(adjacency)
#
# # Step 4: Apply Louvain algorithm for community detection
# louvain = Louvain()
# labels = louvain.fit_predict(adjacency)
#
# # plt.figure(figsize=(8, 6))
# # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap='viridis', edgecolors='k')
# # plt.title("Louvain Clustering on Iris Dataset (Graph-based)")
# # plt.xlabel("Feature 1 (Standardized)")
# # plt.ylabel("Feature 2 (Standardized)")
# # plt.show()