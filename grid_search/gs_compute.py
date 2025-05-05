import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
import itertools
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Import from your existing modules
from gs_constants import DIR_RESULTS, DIR_FIGURES
from gs_algos import load_algorithms_fe, load_algorithms_clust
from gs_datasets import load_all_data
from visualization import scatter_plot


def data_normalisation(X, norm_type=""):
    """Apply different normalization techniques to the data."""
    if norm_type == "":
        return X
    elif norm_type == "minmax":
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler().fit(X)
        X = scaler.transform(X)
        X = np.clip(X, 0, 1)
        return X
    elif norm_type == "standard":
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        return X
    elif norm_type == "robust":
        from sklearn import preprocessing
        scaler = preprocessing.RobustScaler().fit(X)
        X = scaler.transform(X)
        return X
    elif norm_type == "quantile":
        from sklearn import preprocessing
        scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(X)
        X = scaler.transform(X)
        return X
    elif norm_type == "power":
        from sklearn import preprocessing
        scaler = preprocessing.PowerTransformer().fit(X)
        X = scaler.transform(X)
        return X
    else:
        return None


def create_param_grid(algorithm_details):
    """Create a parameter grid for an algorithm."""
    param_grid = {}

    # Convert single values to lists for consistent handling
    for param_name, param_value in algorithm_details["param_grid"].items():
        if not isinstance(param_value, list):
            param_grid[param_name] = [param_value]
        else:
            param_grid[param_name] = param_value

    return param_grid


def evaluate_combination(X, y_true, fe_name, fe_details, fe_params, clust_name, clust_details, clust_params,
                         norm_type, dataset_name):
    """Evaluate a single parameter combination."""
    try:
        # Make sure we're not passing the algorithm name as a parameter
        fe_params_clean = fe_params.copy()
        if "algorithm" in fe_params_clean:
            del fe_params_clean["algorithm"]

        clust_params_clean = clust_params.copy()
        if "algorithm" in clust_params_clean:
            del clust_params_clean["algorithm"]

        # Apply normalization
        X_normalized = data_normalisation(np.copy(X), norm_type=norm_type)

        # Apply feature extraction
        transformer = fe_details["estimator"](**fe_params_clean)
        X_transformed = transformer.fit_transform(X_normalized)

        # Handle special case for SOM output which might be coordinates
        if fe_name == "som":
            if hasattr(X_transformed, 'shape') and X_transformed.shape[1] == 2:
                # SOM returns 2D coordinates which is what we want
                pass
            else:
                print(f"Warning: SOM output shape is unexpected: {X_transformed.shape}")
                return None

        # Ensure we have 2D output for all other algorithms
        elif hasattr(X_transformed, 'shape'):
            if len(X_transformed.shape) == 1:
                # Convert 1D output to 2D with second dimension as zeros
                X_transformed = np.column_stack((X_transformed, np.zeros_like(X_transformed)))
                print(f"Warning: {fe_name} returned 1D output, added zero column")
            elif X_transformed.shape[1] > 2:
                print(f"Warning: {fe_name} returned {X_transformed.shape[1]} features, expected 2")
                # Take only first two dimensions
                X_transformed = X_transformed[:, :2]
        else:
            print(f"Warning: {fe_name} output has no shape attribute")
            return None

        # Apply clustering
        estimator = clust_details["estimator"](**clust_params_clean)
        y_pred = estimator.fit_predict(X_transformed)

        # Calculate metrics
        if len(np.unique(y_pred)) > 1:
            ari = adjusted_rand_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)
            contingency_mat = contingency_matrix(y_true, y_pred)
            purity = np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
            silhouette = silhouette_score(X_normalized, y_pred)
            calinski_harabasz = calinski_harabasz_score(X_normalized, y_pred)
            davies_bouldin = davies_bouldin_score(X_normalized, y_pred)
        else:
            print(f"[1CLUST] {fe_name}, {clust_name}, {fe_params_clean}")
            ari = ami = purity = silhouette = calinski_harabasz = -1
            davies_bouldin = float('inf')  # Worst possible score for DBI

        # Create the result dictionary
        result = {
            "dataset": dataset_name,
            "fe_algorithm": fe_name,
            "clustering_algorithm": clust_name,
            "normalization": norm_type,
            "adjusted_rand_score": ari,
            "adjusted_mutual_info_score": ami,
            "purity_score": purity,
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski_harabasz,
            "davies_bouldin_score": davies_bouldin,
            "norm_davies_bouldin_score": 1 / (1 + davies_bouldin),  # Normalized DBS
            "average_score": (ari + ami + purity + silhouette + (1 / (1 + davies_bouldin))) / 5  # Simple average of normalized metrics
        }

        # Add parameters to result dictionary
        for param_name, param_value in fe_params_clean.items():
            result[f"fe_{param_name}"] = param_value
        for param_name, param_value in clust_params_clean.items():
            result[f"clust_{param_name}"] = param_value

        # Generate and save visualization if this is a good result
        if ari > 0.5 or silhouette > 0.5:  # Only save plots for good results
            try:
                os.makedirs(DIR_FIGURES + "svgs/", exist_ok=True)
                os.makedirs(DIR_FIGURES + "pngs/", exist_ok=True)

                # Create parameter string for filename but handle long values
                fe_param_str = "_".join([f"{k}-{str(v)[:10]}" for k, v in fe_params_clean.items()
                                         if k != "n_components" and not isinstance(v, (list, dict))])
                clust_param_str = "_".join([f"{k}-{str(v)[:10]}" for k, v in clust_params_clean.items()
                                            if k != "n_clusters" and not isinstance(v, (list, dict))])

                # Sanitize filename (remove characters that might cause issues)
                safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
                fe_param_str = ''.join(c if c in safe_chars else '_' for c in fe_param_str)
                clust_param_str = ''.join(c if c in safe_chars else '_' for c in clust_param_str)
                norm_type_safe = ''.join(c if c in safe_chars else '_' for c in norm_type)

                # Generate plot with predicted clusters
                plot_title = f'{fe_name} + {clust_name} on {dataset_name} ({norm_type})'
                scatter_plot.plot(plot_title, X_transformed, y_pred, marker='o')
                filename = f'{dataset_name}_{fe_name}_{clust_name}_{norm_type_safe}'
                plt.savefig(DIR_FIGURES + "svgs/" + filename + '.svg')
                plt.savefig(DIR_FIGURES + "pngs/" + filename + '.png')
                plt.close()

                # Generate plot with ground truth
                plot_title = f'{fe_name} with GT on {dataset_name} ({norm_type})'
                scatter_plot.plot(plot_title, X_transformed, y_true, marker='o')
                filename = f'{dataset_name}_{fe_name}_gt_{norm_type_safe}'
                plt.savefig(DIR_FIGURES + "svgs/" + filename + '.svg')
                plt.savefig(DIR_FIGURES + "pngs/" + filename + '.png')
                plt.close()
            except Exception as e:
                print(f"Error generating plots: {e}")

        return result

    except Exception as e:
        print(f"[ERROR] {fe_name}, {clust_name}, {repr(fe_params)[:50]}, {repr(clust_params)[:50]}, {norm_type}, {e}")

        # Return error result
        result = {
            "dataset": dataset_name,
            "fe_algorithm": fe_name,
            "clustering_algorithm": clust_name,
            "normalization": norm_type,
            "adjusted_rand_score": -1,
            "adjusted_mutual_info_score": -1,
            "purity_score": -1,
            "silhouette_score": -1,
            "calinski_harabasz_score": -1,
            "davies_bouldin_score": float('inf'),
            "norm_davies_bouldin_score": 0,
            "average_score": -1,
            "error": str(e)
        }

        # Add parameters to result dictionary
        for param_name, param_value in fe_params.items():
            result[f"fe_{param_name}"] = param_value
        for param_name, param_value in clust_params.items():
            result[f"clust_{param_name}"] = param_value

        return result

def perform_grid_search(datasets, fe_algorithms, clustering_algorithms,
                        normalizations=["", "minmax", "standard", "robust"], n_jobs=1):
    """
    Perform grid search over feature extraction and clustering algorithms with different parameters
    and normalizations.

    Parameters:
    -----------
    datasets : list of tuples
        List of (name, (X, y)) tuples for each dataset
    fe_algorithms : dict
        Dictionary of feature extraction algorithms
    clustering_algorithms : dict
        Dictionary of clustering algorithms
    normalizations : list
        List of normalization techniques to try
    n_jobs : int
        Number of parallel jobs to run
    """
    # Create directories for results
    os.makedirs(DIR_RESULTS + "grid_search/", exist_ok=True)
    os.makedirs(DIR_RESULTS + "best_params/", exist_ok=True)
    os.makedirs(DIR_RESULTS + "spaces/", exist_ok=True)

    # Dictionary to store best results for each algorithm-dataset combination
    best_results = {}

    # Process each feature extraction algorithm
    for fe_name, fe_details in fe_algorithms.items():
        print(f"Processing feature extraction algorithm: {fe_name}")

        # Ensure n_components is set to 2 for dimensionality reduction
        fe_param_grid = create_param_grid(fe_details)
        if "n_components" in fe_param_grid:
            fe_param_grid["n_components"] = [2]
        elif "n_dims" in fe_param_grid:
            fe_param_grid["n_dims"] = [2]
        elif "n_evecs" in fe_param_grid:
            fe_param_grid["n_evecs"] = [2]

        # Generate all combinations of feature extraction parameters
        fe_param_names = list(fe_param_grid.keys())
        fe_param_values = list(fe_param_grid.values())
        fe_param_combinations = list(itertools.product(*fe_param_values))

        # Process each clustering algorithm
        for clust_name, clust_details in clustering_algorithms.items():
            print(f"  - with clustering algorithm: {clust_name}")

            # Create parameter grid for clustering
            clust_param_grid = create_param_grid(clust_details)

            # Process each dataset
            for dataset_name, (X, y_true) in datasets:
                print(f"    - for dataset: {dataset_name}")

                # Set n_clusters to match ground truth if applicable
                if "n_clusters" in clust_param_grid:
                    clust_param_grid["n_clusters"] = [len(np.unique(y_true))]

                # Generate all combinations of clustering parameters
                clust_param_names = list(clust_param_grid.keys())
                clust_param_values = list(clust_param_grid.values())
                clust_param_combinations = list(itertools.product(*clust_param_values))

                all_results = []

                # Generate tasks for parallel processing
                tasks = []
                for norm_type in normalizations:
                    for fe_params_tuple in fe_param_combinations:
                        fe_params = dict(zip(fe_param_names, fe_params_tuple))

                        for clust_params_tuple in clust_param_combinations:
                            clust_params = dict(zip(clust_param_names, clust_params_tuple))

                            tasks.append((
                                X, y_true, fe_name, fe_details, fe_params,
                                clust_name, clust_details, clust_params,
                                norm_type, dataset_name
                            ))

                # Execute tasks in parallel or sequentially
                if n_jobs != 1:
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(evaluate_combination)(*task) for task in tasks
                    )
                else:
                    results = [evaluate_combination(*task) for task in tasks]

                # Filter out None results
                results = [r for r in results if r is not None]
                all_results.extend(results)

                # Convert to DataFrame and save
                if all_results:
                    df_results = pd.DataFrame(all_results)

                    # Save full grid search results
                    grid_filename = f"{DIR_RESULTS}grid_search/{fe_name}_{clust_name}_{dataset_name}.csv"
                    df_results.to_csv(grid_filename, index=False)

                    # Find best parameter combination for this algorithm-dataset pair
                    # We'll use average_score as our primary metric
                    best_idx = df_results['average_score'].idxmax()
                    best_result = df_results.iloc[best_idx].to_dict()

                    # Store best result
                    key = f"{fe_name}_{clust_name}_{dataset_name}"
                    best_results[key] = best_result

                    # Save best transformed space for this algorithm-dataset pair
                    try:
                        # Extract best parameters - ONLY use parameters from the original param_grid
                        # to avoid including algorithm name or other metadata fields that were added to the results
                        best_norm = best_result['normalization']

                        # Use only the original parameters from fe_details
                        best_fe_params = {}
                        for param_name in fe_details["param_grid"].keys():
                            # The parameter in the results has "fe_" prefix
                            result_key = f"fe_{param_name}"
                            if result_key in best_result:
                                best_fe_params[param_name] = best_result[result_key]

                        # Apply best normalization
                        X_normalized = data_normalisation(np.copy(X), norm_type=best_norm)

                        # Apply best transformation
                        transformer = fe_details["estimator"](**best_fe_params)
                        X_transformed = transformer.fit_transform(X_normalized)

                        # Save transformed space
                        os.makedirs(f"{DIR_RESULTS}spaces/{fe_name}/", exist_ok=True)
                        np.savetxt(f"{DIR_RESULTS}spaces/{fe_name}/{dataset_name}_best.csv",
                                   X_transformed, delimiter=",")
                    except Exception as e:
                        print(f"Error saving best space for {fe_name} on {dataset_name}: {e}")

    # Save all best results as a single dataframe
    best_df = pd.DataFrame(list(best_results.values()))
    best_df.to_csv(f"{DIR_RESULTS}best_params/all_best_params.csv", index=False)

    # Save best results per algorithm
    for fe_name in fe_algorithms.keys():
        fe_best = [v for k, v in best_results.items() if k.startswith(fe_name)]
        if fe_best:
            pd.DataFrame(fe_best).to_csv(
                f"{DIR_RESULTS}best_params/{fe_name}_best_params.csv", index=False)

    # Save best results per dataset
    for dataset_name, _ in datasets:
        dataset_best = [v for k, v in best_results.items() if k.endswith(dataset_name)]
        if dataset_best:
            pd.DataFrame(dataset_best).to_csv(
                f"{DIR_RESULTS}best_params/{dataset_name}_best_params.csv", index=False)

    return best_results


if __name__ == "__main__":
    # Load datasets and algorithms
    datasets = load_all_data()
    fes = load_algorithms_fe()
    clusts = load_algorithms_clust()

    # Define normalization techniques to test
    normalizations = ["", "minmax", "standard", "robust"]

    # Perform grid search
    best_results = perform_grid_search(datasets, fes, clusts, normalizations, n_jobs=4)

    print("Grid search completed. Results saved to:", DIR_RESULTS)