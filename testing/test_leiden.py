# Leiden: Available in leidenalg, usually used with igraph.
import igraph as ig
import leidenalg
from sklearn.metrics import pairwise_distances
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


def remove_largest_eigen_vector(matrix):
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        raise ValueError("The matrix is not symmetric")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Find the index of the largest eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)

    # Select remaining eigenvectors after removing the largest eigenvector
    remaining_eigenvectors = np.delete(eigenvectors, max_eigenvalue_index, axis=1)

    # Construct a new matrix
    new_matrix = remaining_eigenvectors @ np.diag(np.delete(eigenvalues, max_eigenvalue_index)) @ remaining_eigenvectors.T

    return new_matrix



adj_matrix = pairwise_distances(X_transformed)
print(adj_matrix)
adj_matrix = remove_largest_eigen_vector(adj_matrix)
adj_matrix[adj_matrix < 0.5] = 0

# Create an igraph graph object
graph = ig.Graph.Weighted_Adjacency(adj_matrix)
# Apply the Leiden algorithm for community detection evaluating the nº of clusters created by changing the resolution parameter.
for i in np.arange(0.0, 1.05, 0.05):
    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition,
                                         resolution_parameter=i)

    print(i, len(np.unique(partition.membership)))