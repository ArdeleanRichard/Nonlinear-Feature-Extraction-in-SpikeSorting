import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA

from visualization.label_map import LABEL_COLOR_MAP, LABEL_COLOR_MAP2


def plot(title, X, labels=None, plot=True, marker='o', alpha=0.7, binary_markers=None):
    """
    Plots the dataset with or without labels
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    # Handle binary markers efficiently (avoid plotting one point at a time)
    if plot:
        nrDim = len(X[0])
        fig = plt.figure()  # figsize=(16, 12), dpi=400
        plt.title(title)

        if labels is not None:
            try:
                label_color = [LABEL_COLOR_MAP[l] for l in labels]
            except KeyError:
                print('Too many labels! Using default colors...\n')
                label_color = [l for l in labels]
        else:
            label_color = 'gray'

        # Handle binary markers efficiently (avoid plotting one point at a time)
        if binary_markers is not None:
            # Convert binary_markers to numpy array if it's not already
            binary_markers = np.array(binary_markers)

            # Create masks for different marker types
            mask_default = binary_markers == 0
            mask_special = binary_markers == 1

            # Alternate marker for points where binary_markers is 1
            special_marker = 'X' if marker != 'X' else '*'

            if nrDim == 2:
                # Plot points with default marker
                if np.any(mask_default):
                    plt.scatter(X[mask_default, 0], X[mask_default, 1],
                                c=[label_color[i] for i in range(len(label_color)) if mask_default[i]] if isinstance(label_color, list) else label_color,
                                marker=marker, edgecolors='k', alpha=alpha)

                # Plot points with special marker
                if np.any(mask_special):
                    plt.scatter(X[mask_special, 0], X[mask_special, 1],
                                c=[label_color[i] for i in range(len(label_color)) if mask_special[i]] if isinstance(label_color, list) else label_color,
                                marker=special_marker, edgecolors='k', alpha=1)

            elif nrDim == 3:
                plt.axis('off')
                ax = Axes3D(fig)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

                # Plot points with default marker
                if np.any(mask_default):
                    ax.scatter(X[mask_default, 0], X[mask_default, 1], X[mask_default, 2],
                               c=[label_color[i] for i in range(len(label_color)) if mask_default[i]] if isinstance(label_color, list) else label_color,
                               marker=marker, edgecolors='k', s=25, alpha=alpha)

                # Plot points with special marker
                if np.any(mask_special):
                    ax.scatter(X[mask_special, 0], X[mask_special, 1], X[mask_special, 2],
                               c=[label_color[i] for i in range(len(label_color)) if mask_special[i]] if isinstance(label_color, list) else label_color,
                               marker=special_marker, edgecolors='k', s=25, alpha=alpha)

        else:
            # Original plotting without binary markers
            if nrDim == 2:
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k', alpha=alpha)

            elif nrDim == 3:
                plt.axis('off')
                ax = Axes3D(fig)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, edgecolors='k', s=25, alpha=alpha)



def plot2D(title, X, labels=None, plot=True, marker='o', alpha=1, LABEL_COLOR_MAP=LABEL_COLOR_MAP):
    """
    Plots the dataset with or without labels
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    pca_2d = PCA(n_components=2)
    X2D = pca_2d.fit_transform(X)
    if plot:
        fig = plt.figure()  # figsize=(16, 12), dpi=400
        plt.title(title)

        if labels is not None:
            try:
                label_color = [LABEL_COLOR_MAP[l] for l in labels]
            except KeyError:
                print('Too many labels! Using default colors...\n')
                label_color = [l for l in labels]
        else:
            label_color = 'gray'


        plt.scatter(X2D[:, 0], X2D[:, 1], c=label_color, marker=marker, edgecolors='k', alpha=alpha)



def plot_grid(title, X, pn, labels=None, plot=True, marker='o'):
    """
    Plots the dataset with grid
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)
    if plot:
        nrDim = len(X[0])
        label_color = [LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        plt.title(title)
        if nrDim == 2:
            ax = fig.gca()

            ax.set_xticks(np.arange(0, pn, 1))
            ax.set_yticks(np.arange(0, pn, 1))

            plt.scatter(X[:, 0], X[:, 1], marker=marker, c=label_color, s=25, edgecolor='k')
            plt.grid(True)
        if nrDim == 3:
            ax = Axes3D(fig)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.set_xticks(np.arange(0, pn, 1))
            # ax.set_zticks(np.arange(0, pn, 1))
            # ax.set_yticks(np.arange(0, pn, 1))
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, s=25)
            # plt.grid(True)
        # fig.savefig("cevajeg.svg", format='svg', dpi=1200)


if __name__ == "__main__":
    from dataset_parsing import simulations_dataset as ds
    import os
    os.chdir("../")

    for simulation_number in [53, 81, 67, 86]:
        X, y = ds.get_dataset_simulation(simNr=simulation_number)
        plot2D(f"Sim{simulation_number} with PCA and ground truth labels", X, labels=y, plot=True, marker='o', alpha=1, LABEL_COLOR_MAP=LABEL_COLOR_MAP2)
        plt.savefig(f"./paper/figures/fig2_data/Sim{simulation_number}.png")
        plt.savefig(f"./paper/figures/fig2_data/Sim{simulation_number}.svg")
        plt.close()