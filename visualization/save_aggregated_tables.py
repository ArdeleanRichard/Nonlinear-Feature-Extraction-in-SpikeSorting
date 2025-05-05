import pandas as pd
import glob
import os

def read_data():
    columns = ["dataset", "adjusted_rand_score", "adjusted_mutual_info_score", "purity_score", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]

    FOLDER = "../results/saved/"

    methods_dict = {
        'PCA':                  filter_df(f"{FOLDER}pca_kmeans.csv", columns=columns),
        "MDS":                  filter_df(f"{FOLDER}mds_kmeans.csv", columns=columns),
        'ICA':                  filter_df(f"{FOLDER}ica_kmeans.csv", columns=columns),
        'KPCA':                 filter_df(f"{FOLDER}kpca_kmeans.csv", columns=columns),
        'SOM':                  filter_df(f"{FOLDER}som_kmeans.csv", columns=columns),
        # 'AE':                   np.loadtxt(f"{FOLDER}ae_normal.csv", dtype=float, delimiter=","),
        'AE':                   filter_df(f"{FOLDER}ae_kmeans.csv", columns=columns),
        "LLE":                  filter_df(f"{FOLDER}lle_kmeans.csv", columns=columns),
        "MLLE":                 filter_df(f"{FOLDER}mlle_kmeans.csv", columns=columns),
        # "HLLE":               filter_columns_and_save(f"{FOLDER}hlle_kmeans.csv", columns=columns),
        # "LTSA":               filter_columns_and_save(f"{FOLDER}ltsa_kmeans.csv", columns=columns),
        "Keppler Mapper":       filter_df(f"{FOLDER}kmapper_kmeans.csv", columns=columns),
        'Isomap':               filter_df(f"{FOLDER}/isomap_kmeans.csv", columns=columns),
        'Spectral embedding':   filter_df(f"{FOLDER}/spectral_kmeans.csv", columns=columns),
        "t-SNE":                filter_df(f"{FOLDER}tsne_kmeans.csv", columns=columns),
        "Diffusion Map":        filter_df(f"{FOLDER}diffusion_map_kmeans.csv", columns=columns),
        "PHATE":                filter_df(f"{FOLDER}phate_kmeans.csv", columns=columns),
        'UMAP':                 filter_df(f"{FOLDER}umap_kmeans.csv", columns=columns),
        "Trimap":               filter_df(f"{FOLDER}trimap_kmeans.csv", columns=columns),
    }


    dfs = []
    method_names = list(methods_dict.keys())
    for method_name in method_names:
        method_data = methods_dict[method_name]
        dfs.append((method_name, method_data))

    return dfs

def aggregate_by_dataset(sim_nr, dfs):
    target_dataset = f"Sim{sim_nr}"

    rows = []
    for (algorithm, df) in dfs:
        # Filter for the row with the desired dataset
        matching_row = df[df['dataset'] == target_dataset]
        if not matching_row.empty:
            row = matching_row.copy()
            row.insert(0, "algorithm", algorithm)
            rows.append(row)

    # Combine all the rows into a single DataFrame
    if rows:
        result_df = pd.concat(rows, ignore_index=True)

        float_cols = result_df.select_dtypes(include=['float'])
        result_df[float_cols.columns] = float_cols.round(3)

        column_renames = {
            'adjusted_rand_score': 'ARI',
            'adjusted_mutual_info_score': 'AMI',
            'purity_score': 'Purity',
            'silhouette_score': 'SS',
            'calinski_harabasz_score': 'CHS',
            'davies_bouldin_score': 'DBS',
        }
        result_df = result_df.rename(columns=column_renames)

        result_df = result_df.drop(columns=['dataset'])
        result_df.to_csv(f"../paper/tables/sim{sim_nr}.csv", index=False)

        print("Aggregation complete: saved to aggregated_results.csv")
    else:
        print(f"No matching dataset '{target_dataset}' found in the CSVs.")


def filter_df(input_csv, columns):
    df = pd.read_csv(input_csv)

    df_filtered = df[columns]

    return df_filtered

if __name__ == "__main__":
    for simulation_number in [53, 81, 67, 86]:
        dfs = read_data()
        aggregate_by_dataset(simulation_number, dfs)