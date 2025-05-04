import numpy as np
from sklearn import manifold
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from umap import UMAP
from pydiffmap import diffusion_map as dm
from minisom import MiniSom
import phate
import trimap
import kmapper as km
from sklearn.cluster import DBSCAN, KMeans


# for trimap error:
# #Change this:
# # import pkg_resources
# # __version__ = pkg_resources.get_distribution("trimap").version
# #To this:
# from importlib.metadata import version
# __version = version("trimap")


class DiffusionMapWrapper:
    def __init__(self, n_evecs=2, alpha=0.5, **kwargs):
        self.model = dm.DiffusionMap.from_sklearn(n_evecs=n_evecs, alpha=alpha, **kwargs)

    def fit_transform(self, X):
        return self.model.fit_transform(X)


class SOMWrapper:
    def __init__(self, x=10, y=10, input_len=None, sigma=1.0, learning_rate=0.5, num_iteration=1000):
        self.x, self.y = x, y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.input_len = input_len

    def fit(self, X):
        input_len = self.input_len or X.shape[1]
        self.model = MiniSom(x=self.x, y=self.y, input_len=input_len,
                              sigma=self.sigma, learning_rate=self.learning_rate)
        self.model.random_weights_init(X)
        self.model.train_random(X, self.num_iteration)
        return self

    def transform(self, X):
        # map each sample to its best matching unit (row, col)
        winners = [self.model.winner(x) for x in X]
        return np.array(winners)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PHATEWrapper:
    def __init__(self, n_components=2, **kwargs):
        self.op = phate.PHATE(n_components=n_components, **kwargs)

    def fit_transform(self, X):
        return self.op.fit_transform(X)


class TriMapWrapper:
    def __init__(self, n_dims=2, **kwargs):
        self.tri = trimap.TRIMAP(n_dims=n_dims, **kwargs)

    def fit_transform(self, X):
        return self.tri.fit_transform(X)


class KMapperWrapper:
    def __init__(self, verbose=1, n_cubes=10, clusterer_eps=0.5, clusterer_min_samples=5, **kwargs):
        self.mapper = km.KeplerMapper(verbose=verbose)
        self.cover = km.Cover(n_cubes=n_cubes)
        self.clusterer = DBSCAN(eps=clusterer_eps, min_samples=clusterer_min_samples)

    def fit_transform(self, X):
        proj = self.mapper.fit_transform(X, projection=manifold.TSNE)
        # map produces graph but return the low-dim projection
        return proj


def load_algorithms_fe():
    algorithms = {
        # "pca": {
        #     "estimator": PCA,
        #     "param_grid": {
        #         "n_components": 2,
        #     },
        # },
        #
        #
        # "ica": {
        #     "estimator": FastICA,
        #     "param_grid": {
        #         "n_components": 2,
        #         "fun": "logcosh",
        #         "max_iter": 200,
        #         "tol": 1e-3,
        #     },
        # },
        #
        #
        # "isomap": {
        #     "estimator": Isomap,
        #     "param_grid": {
        #         "n_neighbors": 100,
        #         "n_components": 2,
        #         "eigen_solver": "arpack",
        #         "path_method": "D",
        #         "n_jobs": -1,
        #     },
        # },
        #
        # "kpca": {
        #     "estimator": KernelPCA,
        #     "param_grid": {
        #         "n_components": 2,
        #         "kernel": "rbf",
        #         "gamma": 0.1
        #      },
        # },


        # "tsne": {
        #     "estimator": TSNE,
        #     "param_grid": {
        #         "n_components": 2,
        #         "perplexity": 30,
        #         "n_iter": 1000
        #      },
        # },
        #
        # Locally Linear Embedding (LLE) (sklearn.manifold.LocallyLinearEmbedding) - Preserves local linear structures.
        "lle": {
            "estimator": LocallyLinearEmbedding,
            "param_grid": {
                "n_components": 2,
                "n_neighbors": 70,
                "method": "standard"
            },
        },

        "mlle": {
            "estimator": LocallyLinearEmbedding,
            "param_grid": {
                "n_components": 2,
                "n_neighbors": 50,  # n_neighbors > n_components
                "method": "modified"
            },
        },

        # "hlle": {
        #     "estimator": LocallyLinearEmbedding,
        #     "param_grid": {
        #         "n_components": 2,
        #         "n_neighbors": 110,  # n_neighbors > n_components * (n_components + 3) / 2.
        #         "method": "hessian",
        #         "eigen_solver": 'dense', # default arpack Error in determining null-space with ARPACK. Error message: 'Factor is exactly singular'. Note that eigen_solver='arpack' can fail when the weight matrix is singular or otherwise ill-behaved.
        #     },
        # },

        "ltsa": {
            "estimator": LocallyLinearEmbedding,
            "param_grid": {
                "n_components": 2,
                "method": "ltsa",
                "eigen_solver": 'dense',  # default arpack Error in determining null-space with ARPACK. Error message: 'Factor is exactly singular'. Note that eigen_solver='arpack' can fail when the weight matrix is singular or otherwise ill-behaved.
            },
        },

        # # Multidimensional Scaling (MDS) (sklearn.manifold.MDS) - Finds embeddings preserving pairwise distances (metric or non metric).
        # "mds": {
        #     "estimator": MDS,
        #     "param_grid": {
        #         "n_components": 2,
        #         "metric": True
        #     },
        # },

        # Spectral Embedding (Laplacian Eigenmaps, sklearn.manifold.SpectralEmbedding) - Constructs graph Laplacian and uses its eigenvectors for embedding.
        # "spectral": {
        #     "estimator": SpectralEmbedding,
        #     "param_grid": {
        #         "n_components": 2,
        #         "affinity": "nearest_neighbors"
        #     },
        # },







        # "umap": {
        #     "estimator": UMAP,
        #     "param_grid": {
        #         "n_neighbors": 10,
        #         "min_dist": 0.05,
        #         "metric": "chebyshev",
        #         "n_epochs": 500,
        #         "n_components": 2,
        #         "n_jobs": 1,
        #     },
        # },
        #



        # # Diffusion Maps (pydiffmap) - Builds a diffusion operator over the data to reveal intrinsic geometry.
        # "diffusion_map": {
        #     "estimator": DiffusionMapWrapper,
        #     "param_grid": {
        #         "n_evecs": 2,
        #         "alpha": 0.5
        #     },
        # },
    #
    #
    #     "som": {
    #         "estimator": SOMWrapper,
    #         "param_grid": {
    #             "x": 10,
    #             "y": 10,
    #             "sigma": 1.0,
    #             "learning_rate": 0.5,
    #             "num_iteration": 1000
    #         },
    #     },
    #
        # # PHATE (phate) - Heat diffusion based embedding preserving local and global structure, popular in bioinformatics.
        # "phate": {
        #     "estimator": PHATEWrapper,
        #     "param_grid": {
        #         "n_components": 2
        #     },
        # },
    #
        # TriMap (trimap) - Uses triplet constraints (“i closer to j than k”) to optimize embeddings.
    #     "trimap": {
    #         "estimator": TriMapWrapper,
    #         "param_grid": {
    #             "n_dims": 2
    #         },
    #     },
    #
    #     # Kepler Mapper (kmapper) = Topological data analysis Mapper algorithm producing simplicial complexes. kepler-mapper.scikit-tda.org
    #     "kmapper": {
    #         "estimator": KMapperWrapper,
    #         "param_grid": {
    #             "n_cubes": 10,
    #             "clusterer_eps": 0.5,
    #             "clusterer_min_samples": 5
    #         },
    #     },






    }
    return algorithms


def load_algorithms_clust():
    algorithms = {

        "kmeans": {
            "estimator": KMeans,
            "param_grid": {
                "n_clusters": 2,
            },
        },

    }

    return algorithms
