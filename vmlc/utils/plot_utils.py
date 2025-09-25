"""
Utility functions for plotting.
"""

import math
import os
from typing import Any, Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

CMAP = mcolors.ListedColormap(
    [
        "black",
        "red",
        "orange",
        "yellow",
        "limegreen",
        "lightskyblue",
        "magenta",
        "pink",
        "cyan",
        "lightgrey",
    ]
)


def plot_grid(ax: plt.Axes, grid: np.ndarray) -> None:
    """
    Plot a 10x10 matrix with a custom colormap on the provided Axes.

    This function takes a 10x10 matrix of integers, maps the values to specific colors using
    a custom colormap, and displays the matrix on the given Matplotlib Axes.
    Dashed white lines are added at the midpoints to divide the grid into quadrants.

    Args:
        ax (plt.Axes): The Matplotlib Axes on which the grid will be plotted.
        grid (np.ndarray): A 10x10 numpy array of integers representing the grid.

    Returns:
        None
    """
    bounds = np.arange(-0.5, 10.5, 1)
    norm = mcolors.BoundaryNorm(bounds, CMAP.N)
    matrix_colors = np.where(grid == 0, -1, grid)
    ax.imshow(matrix_colors, cmap=CMAP, norm=norm, aspect="equal")

    # Annotate each cell with its value.
    for (i, j), val in np.ndenumerate(grid):
        if val != 0:
            text_color = "white" if val == 0 else "black"
            ax.text(
                j, i, str(val), ha="center", va="center", color=text_color, fontsize=12
            )

    # Add dashed lines for each row (horizontal lines)
    for i in range(1, len(grid)):
        ax.axhline(y=i - 0.5, color="silver", linestyle="-", linewidth=0.5)
    # Add dashed lines for each column (vertical lines)
    for j in range(1, len(grid)):
        ax.axvline(x=j - 0.5, color="silver", linestyle="-", linewidth=0.5)

    ax.axis("off")


def plot_section(
    axes: np.ndarray,
    fig: plt.Figure,
    start_row: int,
    pairs: List[List[np.ndarray]],
    section_title: str,
    max_columns: int = 14,
) -> int:
    """
    Plot a section of input-output grid pairs starting at a given row in the axes grid,
    and draw an arrow from each input grid to the output grid below.

    Args:
        axes (np.ndarray): 2D array of Matplotlib Axes.
        fig (plt.Figure): The Figure object, used to add the arrow annotations.
        start_row (int): Starting row index in the axes grid for this section.
        pairs (List[List[np.ndarray]]): List of tuples (input grid, output grid).
        section_title (str): Prefix for the titles of the plots.
        max_columns (int, optional): Maximum number of columns per row. Defaults to 14.

    Returns:
        int: The next available row index in the axes grid after plotting this section.
    """
    num_pairs = len(pairs)
    rows_needed = 2 * math.ceil(num_pairs / max_columns)

    for i, (input_grid, output_grid) in enumerate(pairs):
        block_row = i // max_columns
        col = i % max_columns
        row_input = start_row + 2 * block_row
        row_output = row_input + 1

        # Plot input and output grids.
        plot_grid(axes[row_input, col], input_grid)
        axes[row_input, col].set_title(f"{section_title} {i + 1}")
        plot_grid(axes[row_output, col], output_grid)

        # Draw an arrow from the bottom center of the input subplot
        # to the top center of the output subplot using ConnectionPatch.
        con = ConnectionPatch(
            xyA=(0.5, 0),
            coordsA=axes[row_input, col].transAxes,
            xyB=(0.5, 1),
            coordsB=axes[row_output, col].transAxes,
            arrowstyle="->",
            color="black",
            lw=2,
        )
        fig.add_artist(con)

    # Turn off any unused subplots in this section.
    total_slots = (rows_needed // 2) * max_columns
    for i in range(num_pairs, total_slots):
        block_row = i // max_columns
        col = i % max_columns
        row_input = start_row + 2 * block_row
        row_output = row_input + 1
        axes[row_input, col].axis("off")
        axes[row_output, col].axis("off")

    return start_row + rows_needed


def plot_sample(
    sample: Dict[str, Any],
    filename: str,
    max_columns: int = 14,
    save_individual: bool = False,
) -> None:
    """
    Plot input and output grids for each section (e.g., 'study_examples' and 'queries')
    from the sample, and save the combined figure to a file. Additionally, if save_individual is True,
    plots each pair of input-output grid separately and saves each as a high-quality PDF for scientific papers.

    Args:
        sample (Dict[str, Any]): Dictionary with sections as keys. Each section (except 'meta_data')
            contains a list of tuples (input grid, output grid) or a dictionary mapping names to such lists.
        filename (str): The filename to save the resulting combined plot.
        max_columns (int, optional): Maximum number of columns per row. Defaults to 14.
        save_individual (bool, optional): If True, also save each input-output pair as a separate high-quality PDF. Defaults to False.

    Returns:
        None
    """
    sections = [key for key in sample.keys() if key != "meta_data"]

    # Compute the number of rows required.
    section_row_counts: Dict[str, int] = {}
    for section in sections:
        section_data = sample[section]
        if isinstance(section_data, dict):
            total_rows_section = 0
            for _, pairs in section_data.items():
                total_rows_section += 2 * math.ceil(len(pairs) / max_columns)
            section_row_counts[section] = total_rows_section
        else:
            section_row_counts[section] = 2 * math.ceil(len(section_data) / max_columns)

    total_rows = sum(section_row_counts.values())
    fig, axes = plt.subplots(
        total_rows, max_columns, figsize=(max_columns * 3, total_rows * 3)
    )
    axes = np.atleast_2d(axes)

    current_row = 0
    for section in sections:
        section_data = sample[section]
        if isinstance(section_data, dict):
            for subkey, pairs in section_data.items():
                current_row = plot_section(
                    axes,
                    fig,
                    current_row,
                    pairs,
                    f"{section.capitalize()} {subkey}",
                    max_columns,
                )
        else:
            current_row = plot_section(
                axes, fig, current_row, section_data, section.capitalize(), max_columns
            )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

    # Save each input and output grid as separate PDFs if requested.
    if save_individual:
        base = os.path.splitext(filename)[0]
        for section in sections:
            section_data = sample[section]
            if isinstance(section_data, dict):
                for subkey, pairs in section_data.items():
                    for i, (input_grid, output_grid) in enumerate(pairs):
                        # Save input grid separately.
                        fig_in, ax_in = plt.subplots(figsize=(3, 3))
                        plot_grid(ax_in, input_grid)
                        plt.tight_layout()
                        in_filename = f"{base}_{section}_{subkey}_input_{i + 1}.pdf"
                        plt.savefig(in_filename, format="pdf", dpi=300)
                        plt.close(fig_in)

                        # Save output grid separately.
                        fig_out, ax_out = plt.subplots(figsize=(3, 3))
                        plot_grid(ax_out, output_grid)
                        plt.tight_layout()
                        out_filename = f"{base}_{section}_{subkey}_output_{i + 1}.pdf"
                        plt.savefig(out_filename, format="pdf", dpi=300)
                        plt.close(fig_out)
            else:
                for i, (input_grid, output_grid) in enumerate(section_data):
                    # Save input grid separately.
                    fig_in, ax_in = plt.subplots(figsize=(3, 3))
                    plot_grid(ax_in, input_grid)
                    plt.tight_layout()
                    in_filename = f"{base}_{section}_input_{i + 1}.pdf"
                    plt.savefig(in_filename, format="pdf", dpi=300)
                    plt.close(fig_in)

                    # Save output grid separately.
                    fig_out, ax_out = plt.subplots(figsize=(3, 3))
                    plot_grid(ax_out, output_grid)
                    plt.tight_layout()
                    out_filename = f"{base}_{section}_output_{i + 1}.pdf"
                    plt.savefig(out_filename, format="pdf", dpi=300)
                    plt.close(fig_out)


def plot_pattern(ax: plt.Axes, matrix: np.ndarray) -> None:
    """
    Plot a 10x10 matrix with custom color mapping.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        matrix (np.ndarray): A 10x10 numpy array to plot.
    """
    cmap = mcolors.ListedColormap(
        [
            "white",
            "black",
            "red",
            "orange",
            "yellow",
            "limegreen",
            "lightskyblue",
            "magenta",
            "pink",
            "cyan",
            "lightgrey",
        ]
    )
    bounds = np.arange(-1.5, 10.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for (i, j), val in np.ndenumerate(matrix):
        color = "white" if val == 0 else "black"
        ax.text(j, i, int(val), ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def display_input_output_plot(patterns: List[List[np.ndarray]], plot_name: str) -> None:
    """
    Display a grid of input and output patterns.

    Args:
        patterns (List[List[np.ndarray]]): A list of lists of 10x10 matrices.
        plot_name (str): The name of the file to save the plot to.
    """
    num_q = len(patterns)
    num_p = len(patterns[0]) if num_q > 0 else 0

    if num_q == 0 or num_p == 0:
        raise ValueError("Patterns list must be a non-empty 2D list of numpy arrays.")

    os.makedirs(os.path.dirname(plot_name), exist_ok=True)

    # Create subplots with correct dimensions
    fig, axs = plt.subplots(
        num_q, num_p, figsize=(max(4, 4 * num_p), max(4, 4 * num_q)), squeeze=False
    )

    for i in range(num_q):
        for j in range(num_p):
            plot_pattern(axs[i, j], patterns[i][j])
            if j == num_p - 3:
                axs[i, j].set_title("query input")
            elif j == num_p - 2:
                axs[i, j].set_title("predicted output")
            elif j == num_p - 1:
                axs[i, j].set_title("target output")

    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()


def plot_individual_grids(grid_pairs: List[List[np.ndarray]], plot_dir: str) -> None:
    """
    Plot individual input and output grids from a list of input/output pairs.

    Args:
        grid_pairs (List[List[np.ndarray]]): A list of pairs of 10x10 input and output grids.
        plot_dir (str): The directory to save the plots to.
    """
    os.makedirs(plot_dir, exist_ok=True)

    for i, (input_grid, output_grid) in enumerate(grid_pairs):
        # Save input grid separately.
        fig_in, ax_in = plt.subplots(figsize=(3, 3))
        plot_grid(ax_in, input_grid)
        plt.tight_layout()
        in_filename = f"input_{i + 1}.pdf"
        plt.savefig(f"{plot_dir}/{in_filename}", format="pdf", dpi=300)
        plt.close(fig_in)

        # Save output grid separately.
        fig_out, ax_out = plt.subplots(figsize=(3, 3))
        plot_grid(ax_out, output_grid)
        plt.tight_layout()
        out_filename = f"output_{i + 1}.pdf"
        plt.savefig(f"{plot_dir}/{out_filename}", format="pdf", dpi=300)
        plt.close(fig_out)
