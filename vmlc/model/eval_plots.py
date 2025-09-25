"""
This file contains code derived and modified from the following source:

Original source:
Lake, B. M. and Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623, 115-121.
<https://github.com/brendenlake/MLC/blob/main/eval.py>

Functions modified from this source:
- `display_console_pred`
- `display_error_pred`

MIT License

Copyright (c) 2022 Brenden Lake

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from wandb.apis import PublicApi

from vmlc.utils.plot_utils import display_input_output_plot
from vmlc.utils.utils import extract, pattern_to_matrix


def compute_average_and_std(
    metrics: Dict[str, List[Tuple[int, float]]], key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the average and standard deviation of a specific metric across steps.

    Args:
        metrics (Dict[str, List[Tuple[int, float]]]): A dictionary of metrics
        where keys are metric names and values are lists of (step, value) tuples.
        key (str): The metric key for which to compute the average and standard deviation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Three numpy arrays:
            - The first array contains the steps.
            - The second array contains the average values of the metric for each step.
            - The third array contains the standard deviation of the metric for each step.
    """
    if key not in metrics:
        print(f"Metric {key} not found.")
        return np.array([]), np.array([]), np.array([])

    values_per_step = defaultdict(list)
    for step, value in metrics[key]:
        values_per_step[step].append(value)

    steps = np.array(sorted(values_per_step.keys()))
    avg = np.array([np.mean(values_per_step[step]) for step in steps])
    std = np.array([np.std(values_per_step[step]) for step in steps])
    return steps, avg, std


def viz_dashboard(metrics: Dict[str, List[Tuple[int, float]]], fig_name: str) -> None:
    """
    Visualize and compare training and validation metrics from extracted TensorBoard metrics.

    Args:
        metrics (Dict[str, List[Tuple[int, float]]]): Dictionary of metrics
        where keys are metric names, and values are lists of (step, value) tuples.
        fig_name (str): Name of the figure file to save the plots.

    Returns:
        None
    """
    # Define metrics to process
    keys = [
        "train_loss_epoch",
        "val_loss",
        "val_acc_novel",
        "val_acc_retrieve",
        "val_color_acc_novel",
        "val_color_acc_retrieve",
        "val_shape_acc_novel",
        "val_shape_acc_retrieve",
    ]

    # Extract steps, averages, and standard deviations for each metric
    stats = {key: compute_average_and_std(metrics, key) for key in keys}

    colors = {
        "train": "tab:blue",
        "val": "tab:green",
        "novel": "tab:red",
        "retrieve": "tab:orange",
    }

    fig_train, axs_train = plt.subplots(1, 1, figsize=(8, 5))
    fig_val, axs_val = plt.subplots(1, 4, figsize=(24, 5))

    accuracy_keys = [
        "val_acc_novel",
        "val_acc_retrieve",
        "val_color_acc_novel",
        "val_color_acc_retrieve",
        "val_shape_acc_novel",
        "val_shape_acc_retrieve",
    ]

    all_acc_values = []
    for key in accuracy_keys:
        _, avg, std = stats[key]
        all_acc_values.extend(avg)
        all_acc_values.extend(avg + std)
        all_acc_values.extend(avg - std)

    y_min = min(all_acc_values) if all_acc_values else 0
    y_max = max(all_acc_values) if all_acc_values else 1
    buffer = (y_max - y_min) * 0.05
    y_min -= buffer
    y_max += buffer

    # Function to plot metrics
    def plot_metric(
        ax, steps, avg_values, std_values, label, color, title, xlabel=None, ylim=None
    ):
        ax.plot(steps, avg_values, label=label, color=color, marker="o")
        ax.fill_between(
            steps,
            avg_values - std_values,
            avg_values + std_values,
            color=color,
            alpha=0.3,
        )
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc="lower right")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Training Loss Plot
    train_steps, avg_train_loss, std_train_loss = stats["train_loss_epoch"]
    if len(train_steps) > 0:
        plot_metric(
            axs_train,
            train_steps,
            avg_train_loss,
            std_train_loss,
            "Train Loss",
            colors["train"],
            "Training Loss",
            xlabel="Steps",
        )

    # Validation Plots
    val_steps, avg_val_loss, std_val_loss = stats["val_loss"]
    if len(val_steps) > 0:
        plot_metric(
            axs_val[0],
            val_steps,
            avg_val_loss,
            std_val_loss,
            "Validation Loss",
            colors["val"],
            "Validation Loss",
            xlabel="Steps",
        )

    val_acc_novel_steps, avg_val_acc_novel, std_val_acc_novel = stats["val_acc_novel"]
    val_acc_retrieve_steps, avg_val_acc_retrieve, std_val_acc_retrieve = stats[
        "val_acc_retrieve"
    ]
    if len(val_acc_novel_steps) > 0 and len(val_acc_retrieve_steps) > 0:
        plot_metric(
            axs_val[1],
            val_acc_novel_steps,
            avg_val_acc_novel,
            std_val_acc_novel,
            "Novel Accuracy",
            colors["novel"],
            "Validation Accuracy",
            xlabel="Steps",
            ylim=(y_min, y_max),
        )
        plot_metric(
            axs_val[1],
            val_acc_retrieve_steps,
            avg_val_acc_retrieve,
            std_val_acc_retrieve,
            "Retrieve Accuracy",
            colors["retrieve"],
            "Validation Accuracy",
            xlabel="Steps",
            ylim=(y_min, y_max),
        )

    color_acc_novel_steps, avg_color_acc_novel, std_color_acc_novel = stats[
        "val_color_acc_novel"
    ]
    color_acc_retrieve_steps, avg_color_acc_retrieve, std_color_acc_retrieve = stats[
        "val_color_acc_retrieve"
    ]
    if len(color_acc_novel_steps) > 0 and len(color_acc_retrieve_steps) > 0:
        plot_metric(
            axs_val[2],
            color_acc_novel_steps,
            avg_color_acc_novel,
            std_color_acc_novel,
            "Color Novel Accuracy",
            colors["novel"],
            "Validation Color Accuracy",
            xlabel="Steps",
            ylim=(y_min, y_max),
        )
        plot_metric(
            axs_val[2],
            color_acc_retrieve_steps,
            avg_color_acc_retrieve,
            std_color_acc_retrieve,
            "Color Retrieve Accuracy",
            colors["retrieve"],
            "Validation Color Accuracy",
            xlabel="Steps",
            ylim=(y_min, y_max),
        )

    shape_acc_novel_steps, avg_shape_acc_novel, std_shape_acc_novel = stats[
        "val_shape_acc_novel"
    ]
    shape_acc_retrieve_steps, avg_shape_acc_retrieve, std_shape_acc_retrieve = stats[
        "val_shape_acc_retrieve"
    ]
    if len(shape_acc_novel_steps) > 0 and len(shape_acc_retrieve_steps) > 0:
        plot_metric(
            axs_val[3],
            shape_acc_novel_steps,
            avg_shape_acc_novel,
            std_shape_acc_novel,
            "Shape Novel Accuracy",
            colors["novel"],
            "Validation Shape Accuracy",
            xlabel="Steps",
            ylim=(y_min, y_max),
        )
        plot_metric(
            axs_val[3],
            shape_acc_retrieve_steps,
            avg_shape_acc_retrieve,
            std_shape_acc_retrieve,
            "Shape Retrieve Accuracy",
            colors["retrieve"],
            "Validation Shape Accuracy",
            xlabel="Steps",
            ylim=(y_min, y_max),
        )

    fig_train.tight_layout()
    fig_val.tight_layout()

    fig_train.savefig(f"train_{fig_name}")
    fig_val.savefig(f"val_{fig_name}")
    plt.show()
    print(f"Plots saved as train_{fig_name} and val_{fig_name}")


def extract_metrics_from_tensorboard(
    root_dir: str,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract metrics from all TensorBoard logs in subdirectories.

    Args:
        root_dir (str): Root directory containing subdirectories with TensorBoard logs.

    Returns:
        dict: Dictionary of metrics aggregated from all logs.
        Keys are metric names, values are lists of (step, value) tuples.
    """
    metrics = defaultdict(list)

    # Traverse subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):  # TensorBoard event file
                event_path = os.path.join(subdir, file)
                print(f"Processing: {event_path}")

                # Load event file
                ea = event_accumulator.EventAccumulator(event_path)
                ea.Reload()

                # Extract scalars
                for tag in ea.Tags()["scalars"]:
                    scalars = ea.Scalars(tag)
                    for scalar in scalars:
                        metrics[tag].append((scalar.step, scalar.value))

    return dict(metrics)


def extract_metrics_from_wandb(project_name: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Extract metrics from all runs in a WandB project.

    Args:
        project_name (str): The WandB project name.

    Returns:
        dict: Dictionary of metrics aggregated across all runs.
              Keys are metric names, values are lists of (step, value) tuples.
    """
    api = PublicApi()
    metrics = defaultdict(list)

    # Fetch all runs in the project
    runs = api.runs(project_name)

    for run in runs:
        print(f"Processing run: {run.name} ({run.id})")
        history = run.history(keys=None, pandas=False)  # Fetch all metrics for the run

        for record in history:
            step = record["_step"]
            for metric, value in record.items():
                if metric not in [
                    "_step",
                    "_runtime",
                    "_timestamp",
                ]:  # Exclude WandB-specific keys
                    metrics[metric].append((step, value))

    return dict(metrics)


def display_console_pred(
    samples_pred: List[Dict[str, Any]], plot_name: str, with_copy_task: bool = True
) -> None:
    """
    Display and plot a sample of predictions from the model.

    This function randomly selects 20 samples from the provided predictions, organizes them into patterns,
    and then plots these patterns using the `display_input_output_plot` function.

    Args:
        samples_pred (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains the model's predictions and related data for a single sample.
        plot_name (str): The name of the file to save the plot.
        with_copy_task (bool): Training with/without copy tasks.

    Returns:
        None
    """
    enumerated_samples = list(enumerate(samples_pred))
    selected_samples = random.sample(enumerated_samples, 5)
    patterns = []
    for _, sample in selected_samples:
        if with_copy_task:
            copy_pattern = []
            novel_pattern = []
            for s in range(len(sample["xs"])):
                copy_pattern.append(sample["xs"][s])
                copy_pattern.append(sample["ys"][s])
                novel_pattern.append(sample["xs"][s])
                novel_pattern.append(sample["ys"][s])
            in_support = sample["in_support"]
            is_novel = np.logical_not(in_support)
            copy_pattern.append(np.array(extract(in_support, sample["xq"])[0]))
            copy_pattern.append(
                pattern_to_matrix(extract(in_support, sample["yq_predict"])[0])
            )
            copy_pattern.append(np.array(extract(in_support, sample["yq"])[0]))
            novel_pattern.append(np.array(extract(is_novel, sample["xq"])[0]))
            novel_pattern.append(
                pattern_to_matrix(extract(is_novel, sample["yq_predict"])[0])
            )
            novel_pattern.append(np.array(extract(is_novel, sample["yq"])[0]))
            patterns.extend([copy_pattern, novel_pattern])

        else:  # without copy tasks
            novel_pattern = []
            for s in range(len(sample["xs"])):
                novel_pattern.append(sample["xs"][s])
                novel_pattern.append(sample["ys"][s])
            in_support = sample["in_support"]
            is_novel = np.logical_not(in_support)
            novel_pattern.append(np.array(extract(is_novel, sample["xq"])[0]))
            novel_pattern.append(
                pattern_to_matrix(extract(is_novel, sample["yq_predict"])[0])
            )
            novel_pattern.append(np.array(extract(is_novel, sample["yq"])[0]))
            patterns.extend([novel_pattern])

    if len(patterns) > 0:
        display_input_output_plot(patterns, plot_name)


def display_error_pred(samples_pred: List[Dict[str, Any]], plot_name: str) -> None:
    """
    Display and plot predictions where the model made errors.

    This function randomly selects 50 samples from the provided predictions, identifies errors in the "novel" predictions,
    and organizes them into patterns. These patterns are then plotted using the `display_input_output_plot` function.

    Args:
        samples_pred (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains the model's predictions and related data for a single sample.
        plot_name (str): The name of the file to save the plot.

    Returns:
        None
    """
    enumerated_samples = list(enumerate(samples_pred))
    selected_samples = random.sample(enumerated_samples, 10)
    patterns = []
    for _, sample in selected_samples:
        novel_pattern = []
        for s in range(len(sample["xs"])):
            novel_pattern.append(sample["xs"][s])
            novel_pattern.append(sample["ys"][s])
        in_support = sample["in_support"]
        is_novel = np.logical_not(in_support)
        if not np.all(
            pattern_to_matrix(extract(is_novel, sample["yq_predict"])[0])
            == extract(is_novel, sample["yq"])[0]
        ):
            novel_pattern.append(np.array(extract(is_novel, sample["xq"])[0]))
            novel_pattern.append(
                pattern_to_matrix(extract(is_novel, sample["yq_predict"])[0])
            )
            novel_pattern.append(np.array(extract(is_novel, sample["yq"])[0]))
            patterns.append(novel_pattern)

    if len(patterns) > 0:
        display_input_output_plot(patterns, plot_name)
