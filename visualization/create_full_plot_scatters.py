import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import figure
import os
import numpy as np
from io import BytesIO
import re


def create_scatter_grid(plot_map, file_dir, output_path=None, file_format="png",
                        figsize=(15, 10), dpi=100, row_layout=None, hspace=0.1, title_height_percent=0, y=1.0):
    """
    Create a grid of scatter plots from individual files based on a mapping.

    Parameters:
    -----------
    plot_map : dict
        A dictionary mapping (row, col) positions to algorithm names.
        Example: {(0, 0): 'algorithm1', (0, 1): 'algorithm2', ...}
    file_dir : str
        Directory containing the scatter plot files.
    output_path : str, optional
        Path to save the combined figure. If None, the figure is just displayed.
    file_format : str, optional
        Format of the input files ('png' or 'svg'). Default is 'png'.
    figsize : tuple, optional
        Size of the output figure in inches. Default is (15, 10).
    dpi : int, optional
        DPI of the output figure. Default is 100.
    row_layout : list, optional
        Number of plots in each row. Example: [3, 3, 5, 5] for 4 rows with varying columns.
        If None, a uniform grid shape will be assumed.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The combined figure object.
    """
    if row_layout is None:
        # Default to a uniform grid if row_layout is not specified
        max_cols = max([pos[1] for pos in plot_map.keys()]) + 1
        max_rows = max([pos[0] for pos in plot_map.keys()]) + 1
        row_layout = [max_cols] * max_rows

    # Create figure with the correct number of rows
    num_rows = len(row_layout)
    max_cols = max(row_layout)

    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Calculate relative heights for each row (based on number of plots in that row)
    row_heights = [1] * num_rows

    # Create GridSpec with custom row heights and minimal spacing
    gs = fig.add_gridspec(nrows=num_rows, ncols=max_cols, height_ratios=row_heights,
                          hspace=hspace, wspace=0.01)

    # Store loaded images to avoid reloading
    loaded_images = {}

    # Create axes for each position in plot_map
    all_axes = {}
    for row in range(num_rows):
        for col in range(row_layout[row]):
            ax = fig.add_subplot(gs[row, col])
            all_axes[(row, col)] = ax
            ax.axis('off')

    # Fill in the specified positions
    for position, algorithm in plot_map.items():
        row, col = position
        if (row, col) in all_axes:
            ax = all_axes[(row, col)]

            # Match any prefix before algorithm name using regex pattern
            filename_pattern = f".*_{algorithm}_*"

            # Look for matching files in the directory
            file_path = None
            for file in os.listdir(file_dir):
                if re.match(filename_pattern, file) and file.endswith(f".{file_format}"):
                    file_path = os.path.join(file_dir, file)
                    break

            if file_path:
                if file_format == "png":
                    img = mpimg.imread(file_path)
                    if title_height_percent != 0:
                        title_height = int(img.shape[0] * (title_height_percent / 100))
                        img = img[title_height:, :, :]
                    ax.imshow(img)

                # Add algorithm name as title
                ax.set_title(title_map[algorithm],  y=y)
            else:
                ax.text(0.5, 0.5, f"Missing: {algorithm}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)

    # Adjust subplot parameters for extremely tight spacing
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=hspace)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Grid saved to {output_path}")

    return fig

title_map = {
    "pca": "PCA",
    "mds": "MDS",
    "ica": "ICA",
    "kpca": "Kernel PCA",
    "som": "SOM",
    "ae": "AE",
    "lle": "LLE",
    "mlle": "MLLE",
    "kmapper": "Keppler Mapper",
    "isomap": "Isomap",
    "spectral": "Spectral embedding",
    "diffusion_map": "Diffusion Map",
    "tsne": "t-SNE",
    "phate": "PHATE",
    "umap": "UMAP",
    "trimap": "TriMap",
    "ARI": "ARI",
    "AMI": "AMI",
    "Purity": "Purity",
    "SS": "SS",
    "CHS": "CHS",
    "DBS": "DBS",
}

def main_scatter_plots(scatter_folders=["fig3_Sim53", "fig4_Sim81", "fig5_Sim67", "fig6_Sim86"]):
    # position_to_algorithm = {
    #     (0, 0): "pca", (0, 1): "mds", (0, 2): "ica",
    #     (1, 0): "kpca", (1, 1): "som", (1, 2): "ae",
    #     (2, 0): "lle", (2, 1): "mlle", (2, 2): "kmapper",
    #     (3, 0): "isomap", (3, 1): "spectral", (3, 2): "tsne",
    #     (4, 0): "diffusion_map", (4, 1): "phate", (4, 2): "umap",
    #     (5, 1): "trimap"
    # }

    position_to_algorithm = {
        (0, 0): "pca", (0, 1): "mds", (0, 2): "ica",
        (1, 0): "kpca", (1, 1): "som", (1, 2): "ae",
        (2, 0): "lle", (2, 1): "mlle", (2, 2): "isomap",
        (3, 0): "spectral", (3, 1): "tsne", (3, 2): "diffusion_map",
        (4, 0): "phate", (4, 1): "umap", (4, 2): "trimap"
    }

    # Directory where your scatter plot files are located
    fig_folder = "../paper/figures/"
    for scatter_plot_folder in scatter_folders:

        # Create the grid
        fig = create_scatter_grid(
            plot_map=position_to_algorithm,
            file_dir=fig_folder + scatter_plot_folder,
            output_path=fig_folder + f"{scatter_plot_folder}.png",
            file_format="png",  # or "svg"
            figsize=(8, 10),
            dpi=600,
            hspace=0.1,
            title_height_percent=12
        )

        plt.close()


def main_metrics():
    # Define the mapping of positions to algorithms
    # This is just an example - you'll need to customize this
    position_to_algorithm = {
        (0, 0): "ARI", (0, 1): "AMI",
        (1, 0): "Purity", (1, 1): "SS",
        (2, 0): "CHS", (2, 1): "DBS",
    }

    # Directory where your scatter plot files are located
    fig_folder = "../paper/figures/"

    for scatter_plot_folder in ["fig7_box"] :

        # Create the grid
        fig = create_scatter_grid(
            plot_map=position_to_algorithm,
            file_dir=fig_folder + scatter_plot_folder,
            output_path=fig_folder + f"{scatter_plot_folder}.png",
            file_format="png",  # or "svg"
            figsize=(10, 8),
            dpi=600,
            hspace=0.01,
            title_height_percent=10,
            y=1,
        )

        plt.close()


    for scatter_plot_folder in ["fig8_ttest"] :

        # Create the grid
        fig = create_scatter_grid(
            plot_map=position_to_algorithm,
            file_dir=fig_folder + scatter_plot_folder,
            output_path=fig_folder + f"{scatter_plot_folder}.png",
            file_format="png",  # or "svg"
            figsize=(6, 10),
            dpi=600,
            hspace=0.01,
            y=0.9,
        )

        plt.close()


# Example usage:
if __name__ == "__main__":
    # main_scatter_plots(scatter_folders=["fig3_Sim53", "fig4_Sim81", "fig5_Sim67", "fig6_Sim86"])
    # main_metrics()
    main_scatter_plots(scatter_folders=["fig9_kampff_c28", "fig10_kampff_c37"])