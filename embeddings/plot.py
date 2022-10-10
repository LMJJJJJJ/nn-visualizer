from typing import Union, List, Dict
import os
import os.path as osp
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.lines import Line2D
from .arrow3d import *


def plot_sample_emb_3d(
        embeddings: Union[np.ndarray, torch.Tensor],
        class_directions: Union[np.ndarray, torch.Tensor, None],
        labels: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        colors: List[str] = None,
        save_path: str = "sample_emb_3d.png"
):
    '''
    This function plots the sample embeddings in a scatter plot, with each class in a different color.
    :param embeddings: the embeddings of each sample, with shape [N, d']
    :param labels: the GROUND-TRUTH labels of each sample, with shape [N, ]
    :param class_names: the name of each category, this will be shown in visualization
    :param cmaps: the color of each category
    :param save_path: the save path
    :return: void
    '''

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(class_directions, torch.Tensor):
        class_directions = class_directions.detach().cpu().numpy()

    if colors is None and len(class_names) == 10:
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    else:
        raise NotImplementedError

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(False)
    coord_range = 3.0
    # make the panes transparent
    ax.set_xlim(-coord_range, coord_range)
    ax.set_ylim(-coord_range, coord_range)
    ax.set_zlim(-coord_range, coord_range)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.zaxis.set_major_locator(MultipleLocator(1.0))

    for i in range(len(class_names)):
        indices = (labels == i)
        X = embeddings[indices]
        if class_directions is not None:
            ax.arrow3D(
                0, 0, 0, *(class_directions[i] * coord_range * 0.5),
                mutation_scale=10, fc=colors[i], ec=colors[i]
            )
        ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            s=25, color=colors[i], depthshade=False,
            alpha=0.5, edgecolors=None, linewidths=0
        )
        ax.scatter([], [], [], color=colors[i], s=20, label=class_names[i])  # generate the legend

    ax.view_init(elev=5, azim=-60)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close("all")


def plot_curves(
        plot_dict: Dict[str, List[float]],
        save_folder: str, save_name: str = "",
        xlabels: List = None,
        ylabels: List = None,
):
    '''
    Plot curves for each key in the plot_dict dictionary.
    :param plot_dict: field -> plot list
    :param save_folder: the save folder
    :param save_name: to specify the save name
    :return: void
    '''
    os.makedirs(save_folder, exist_ok=True)
    if len(save_name) > 0: save_name = f"{save_name}_"
    for i, field in enumerate(plot_dict.keys()):
        plt.figure(figsize=(6, 6))
        plt.title(f"{field}'s curve")
        X = np.arange(1, len(plot_dict[field]) + 1)
        plt.plot(X, plot_dict[field])
        if xlabels is not None:
            plt.xlabel(xlabels[i])
        if ylabels is not None:
            plt.ylabel(ylabels[i])
        plt.grid()
        plt.tight_layout()
        plt.savefig(osp.join(save_folder, f"{save_name}{field}.png"), dpi=200)


def _get_selected_region_mask(sqrt_grid_num, selected_regions_id):
    if isinstance(selected_regions_id, int): selected_regions_id = [selected_regions_id]
    mask = np.zeros(shape=(sqrt_grid_num, sqrt_grid_num))
    for region_id in selected_regions_id:
        row_id = region_id // sqrt_grid_num
        column_id = region_id % sqrt_grid_num
        mask[row_id, column_id] = 1
    return mask


def _plot_selected_region_boundary(ax, selected_regions_mask, sqrt_grid_num, grid_width, linecolor, linewidth):
    for i in range(selected_regions_mask.shape[0]):
        for j in range(selected_regions_mask.shape[1]):
            if selected_regions_mask[i, j] == 0: continue
            # edge on the left
            if (j != 0 and selected_regions_mask[i, j-1] == 0) or j == 0:
                line = Line2D(
                    xdata=[j*grid_width - 0.5, j*grid_width - 0.5],
                    ydata=[i*grid_width - 0.5, (i+1)*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
            # edge on the right
            if (j != sqrt_grid_num-1 and selected_regions_mask[i, j+1] == 0) or j == sqrt_grid_num-1:
                line = Line2D(
                    xdata=[(j+1)*grid_width - 0.5, (j+1)*grid_width - 0.5],
                    ydata=[i*grid_width - 0.5, (i+1)*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
            # edge on the top
            if (i != 0 and selected_regions_mask[i-1, j] == 0) or i == 0:
                line = Line2D(
                    xdata=[j*grid_width - 0.5, (j+1)*grid_width - 0.5],
                    ydata=[i*grid_width - 0.5, i*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)
            # edge on the bottom
            if (i != sqrt_grid_num-1 and selected_regions_mask[i+1, j] == 0) or i == sqrt_grid_num-1:
                line = Line2D(
                    xdata=[j*grid_width - 0.5, (j+1)*grid_width - 0.5],
                    ydata=[(i+1)*grid_width - 0.5, (i+1)*grid_width - 0.5],
                    color=linecolor, linewidth=linewidth
                )
                ax.add_line(line)


def ax_highlight_regions(ax, image, grid_width, region_ids, alpha=0.5, overlay_color=(0, 0, 0), linecolor="red", linewidth=4):
    image_width = image.shape[2]
    sqrt_grid_num = int(np.ceil(image_width / grid_width))
    selected_regions_mask = _get_selected_region_mask(sqrt_grid_num, region_ids)

    # masked image
    overlay_color = np.array(overlay_color).reshape((-1, 1, 1))
    image_ = image.copy()
    for i in range(selected_regions_mask.shape[0]):
        for j in range(selected_regions_mask.shape[1]):
            if selected_regions_mask[i, j] == 0:
                image_[:, i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] *= (1 - alpha)
                image_[:, i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] += overlay_color * alpha

    ax.imshow(image_.transpose(1, 2, 0).clip(0, 1))
    plt.axis("off")
    _plot_selected_region_boundary(
        ax, selected_regions_mask, sqrt_grid_num, grid_width,
        linecolor=linecolor, linewidth=linewidth
    )
