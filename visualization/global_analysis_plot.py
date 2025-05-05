import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy import stats

from constants import LABEL_COLOR_MAP_SMALLER
import seaborn as sn

os.chdir("../")

def plot_box(title, data, method_names, conditions):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    c = 'k'
    black_dict = {  # 'patch_artist': True,
        # 'boxprops': dict(color=c, facecolor=c),
        # 'capprops': dict(color=c),
        # 'flierprops': dict(color=c, markeredgecolor=c),
        'medianprops': dict(color=c),
        # 'whiskerprops': dict(color=c)
    }

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False, **black_dict)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'{title} for all 95 simulations',
        xlabel='Feature Extraction Method',
        ylabel='Performance',
    )

    # Now fill the boxes with desired colors
    num_boxes = len(data)

    for i in range(num_boxes):

        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        med = bp['medians'][i]

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=LABEL_COLOR_MAP_SMALLER[i % len(method_names)]))

        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 1.1
    # bottom = 0
    # ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(method_names, len(conditions)), rotation=0, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(num_boxes) + 1
    # for id, (method, y) in enumerate(zip(METHODS, np.arange(0.01, 0.03 * len(METHODS), 0.03).tolist())):
    #     fig.text(0.90, y, METHODS[id],
    #              backgroundcolor=LABEL_COLOR_MAP2[id],
    #              color='black', weight='roman', size='x-small')

    plt.savefig(f"./figures/global/boxplot_{title}_global_analysis.svg")
    plt.savefig(f"./figures/global/boxplot_{title}_global_analysis.png")
    plt.close()


def filter_columns_and_save(input_csv, columns):
    df = pd.read_csv(input_csv)

    df_filtered = df[columns]

    base_name, ext = os.path.splitext(input_csv)
    output_csv = f"{base_name}_simple{ext}"

    df_filtered.to_csv(output_csv, index=False, header=False)

    return df_filtered.to_numpy()



def compute_ttest(data, method_names):
    ttest_matrix = np.zeros((len(method_names), len(method_names)), dtype=float)
    labels = np.zeros((len(method_names), len(method_names)), dtype=object)
    for m1_id, m1 in enumerate(method_names):
        for m2_id, m2 in enumerate(method_names):
            result = stats.ttest_ind(data[m1_id], data[m2_id], equal_var=True)[1] * (len(method_names) * (len(method_names) - 1) / 2)
            if result > 0.05:
                ttest_matrix[m1_id][m2_id] = -1
                labels[m1_id][m2_id] = ""
            elif 0.01 < result < 0.05:
                ttest_matrix[m1_id][m2_id] = 0
                labels[m1_id][m2_id] = "*"
            else:
                ttest_matrix[m1_id][m2_id] = 1
                labels[m1_id][m2_id] = f"**"

    return ttest_matrix, labels


def plot_ttest_matrix(metric_name, method_names, ttest_matrix, labels):
    df_cm = pd.DataFrame(ttest_matrix, index=method_names, columns=method_names)
    plt.figure(figsize=(11, 11))
    pallete = sn.color_palette("magma", as_cmap=True)
    sn.heatmap(df_cm, annot=False, fmt="", cmap=pallete)
    sn.heatmap(df_cm, annot=labels, annot_kws={'va': 'top', 'size': 14}, fmt="s", cbar=False, cmap=pallete, linewidths=5e-3, linecolor='gray')
    plt.savefig(f'./figures/global/confusion_{metric_name}_global_analysis.svg')
    plt.close()


def main(methods_dict):
    for metric_id, metric_name in enumerate(metric_names):
        data = []
        method_names = list(methods_dict.keys())
        for method_name in method_names:
            method_data = methods_dict[method_name]
            data.append(method_data[:, metric_id].tolist())

        # np.savetxt(f"./figures/global/ttest_{metric_name}.csv", np.array(ttest_matrix), delimiter=",")

        # T-TESTING
        ttest_matrix, labels = compute_ttest(data, method_names)
        plot_ttest_matrix(metric_name, method_names, ttest_matrix, labels)

        plot_box(metric_name, data, method_names, [metric_name])


if __name__ == "__main__":
    # pca = filter_columns_and_save(f"./results/pca_kmeans.csv", columns=columns)
    # ica = filter_columns_and_save(f"./results/ica_kmeans.csv", columns=columns)
    # isomap = filter_columns_and_save(f"./results/spaces/isomap_kmeans.csv", columns=columns)
    # umap = filter_columns_and_save(f"./results/umap_kmeans.csv", columns=columns)
    # ae_normal = np.loadtxt(f"./results/ae_normal.csv", dtype=float, delimiter=",")
    # tsne = filter_columns_and_save(f"./results/tsne_kmeans.csv", columns=columns)
    # lle = filter_columns_and_save(f"./results/lle_kmeans.csv", columns=columns)
    # kpca = filter_columns_and_save(f"./results/kpca_kmeans.csv", columns=columns)
    # trimap = filter_columns_and_save(f"./results/trimap_kmeans.csv", columns=columns)
    # kmapper = filter_columns_and_save(f"./results/kmapper_kmeans.csv", columns=columns)
    # mds = filter_columns_and_save(f"./results/mds_kmeans.csv", columns=columns)

    # pca =               np.loadtxt(f"./results/pca.csv", dtype=float, delimiter=",")
    # ica =               np.loadtxt(f"./results/ica.csv", dtype=float, delimiter=",")
    # isomap =            np.loadtxt(f"./results/isomap.csv", dtype=float, delimiter=",")
    # ae_normal =         np.loadtxt(f"./results/ae_normal.csv", dtype=float, delimiter=",")
    # vade =              np.loadtxt(f"./results/vade.csv", dtype=float, delimiter=",")
    # metric_names = ['ARI', 'AMI', 'Purity', 'DBS', 'CHS', 'SS']

    columns = ["adjusted_rand_score", "adjusted_mutual_info_score", "purity_score", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
    metric_names = ['ARI', 'AMI', 'Purity', 'SS', 'CHS', 'DBS']

    FOLDER = "./results/saved/"
    methods_dict = {
        'PCA':                  filter_columns_and_save(f"{FOLDER}pca_kmeans.csv", columns=columns),
        "MDS":                  filter_columns_and_save(f"{FOLDER}mds_kmeans.csv", columns=columns),
        'ICA':                  filter_columns_and_save(f"{FOLDER}ica_kmeans.csv", columns=columns),
        'KPCA':                 filter_columns_and_save(f"{FOLDER}kpca_kmeans.csv", columns=columns),
        'SOM':                  filter_columns_and_save(f"{FOLDER}som_kmeans.csv", columns=columns),
        'Isomap':               filter_columns_and_save(f"{FOLDER}/isomap_kmeans.csv", columns=columns),
        "t-SNE":                filter_columns_and_save(f"{FOLDER}tsne_kmeans.csv", columns=columns),
        'AE':                   np.loadtxt(f"{FOLDER}ae_normal.csv", dtype=float, delimiter=","),
        "LLE":                  filter_columns_and_save(f"{FOLDER}lle_kmeans.csv", columns=columns),
        "MLLE":                 filter_columns_and_save(f"{FOLDER}mlle_kmeans.csv", columns=columns),
        # "HLLE":               filter_columns_and_save(f"{FOLDER}hlle_kmeans.csv", columns=columns),
        # "LTSA":               filter_columns_and_save(f"{FOLDER}ltsa_kmeans.csv", columns=columns),
        "Keppler Mapper":       filter_columns_and_save(f"{FOLDER}kmapper_kmeans.csv", columns=columns),
        "Diffusion Map":        filter_columns_and_save(f"{FOLDER}diffusion_map_kmeans.csv", columns=columns),
        "PHATE":                filter_columns_and_save(f"{FOLDER}phate_kmeans.csv", columns=columns),
        'UMAP':                 filter_columns_and_save(f"{FOLDER}umap_kmeans.csv", columns=columns),
        "Trimap":               filter_columns_and_save(f"{FOLDER}trimap_kmeans.csv", columns=columns),
    }

    main(methods_dict)