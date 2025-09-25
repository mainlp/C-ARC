"""
This file contains code derived and modified from the following source:

Original source:
Lake, B. M. and Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623, 115-121.
<https://github.com/brendenlake/MLC/blob/main/eval.py>

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

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vmlc.data_prep.base_tokenizer import BaseTokenizer
from vmlc.utils.utils import extract, flatten_matrices, pattern_to_matrix
from vmlc.utils.vgrammar_utils import coordinates2shape, find_groups


def evaluate_ll(net, batch):
    """
    Evaluate the log-likelihood for a single batch.

    Args:
        net (torch.nn.Module): The model to evaluate.
        batch (dict): A dictionary containing the batch data. It should include:
            - "yq_padded" (torch.Tensor): The padded target sequences for the batch.
            - "yq_lengths" (torch.Tensor): The lengths of each target sequence in the batch.

    Returns:
        dict: A dictionary containing the evaluation results, including:
            - "ll_by_cell" (float): The log-likelihood per cell.
            - "N" (float): The total number of target elements across all sequences in the batch.
            - "ll" (float): The total log-likelihood for the batch.
    """
    target_batches = batch["yq_eos"]
    batch_size = target_batches.size(0)
    target_length = target_batches.size(1)

    decoder_output = net(batch)
    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1])
    loss = net.loss_fn(logits_flat, target_batches.reshape(-1))
    loglike = -loss.item()

    return {
        "ll_by_cell": loglike,
        "N": float(batch_size * target_length),
        "ll": loglike * float(batch_size * target_length),
    }


def extract_episode_accuracy(
    list_samples: List[Dict[str, Any]],
    in_support: np.ndarray,
    model_predictions: np.ndarray,
    acc_dict: Dict[str, np.ndarray],
    q_idx: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts accuracy metrics for each episode from the given samples and accuracy dictionary.

    Args:
        list_samples (List[Dict[str, Any]]): A list of sample dictionaries for each episode.
        study_examples (List[Dict[str, Any]]): A list of study example dictionaries for each episode.
        acc_dict (Dict[str, np.ndarray]): A dictionary containing arrays of accuracy metrics, such as 'exact_match_acc', 'color_acc', and 'shape_acc'.
        q_idx (np.ndarray): An array of indices mapping queries to their respective episodes.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
            - A list of dictionaries with extracted accuracy metrics for each episode.
            - A list of dictionaries with updated sample information for each episode, including 'yq_predict' and 'in_support'.
    """
    episode_metrics: List[Dict[str, Any]] = []
    episode_samples: List[Dict[str, Any]] = copy.deepcopy(list_samples)

    for batch_idx in range(len(episode_samples)):
        yq_sel = q_idx == batch_idx  # select for queries in this episode

        in_support_sel = in_support[yq_sel]
        in_query = np.logical_not(in_support_sel)

        # compute metrics
        full_episode_acc = acc_dict["exact_match_acc"][yq_sel]
        full_episode_color_acc = acc_dict["color_acc"][yq_sel]
        full_episode_shape_acc = acc_dict["shape_acc"][yq_sel]

        metric: Dict[str, Any] = {
            "full_episode_acc": (
                np.mean(full_episode_acc) if len(full_episode_acc) > 0 else np.nan
            ),
            "full_episode_color_acc": (
                np.mean(full_episode_color_acc)
                if len(full_episode_color_acc) > 0
                else np.nan
            ),
            "full_episode_shape_acc": (
                np.mean(full_episode_shape_acc)
                if len(full_episode_shape_acc) > 0
                else np.nan
            ),
            "copy_acc": (
                np.mean(full_episode_acc[in_support_sel])
                if len(full_episode_acc[in_support_sel]) > 0
                else np.nan
            ),
            "copy_color_acc": (
                np.mean(full_episode_color_acc[in_support_sel])
                if len(full_episode_color_acc[in_support_sel]) > 0
                else np.nan
            ),
            "copy_shape_acc": (
                np.mean(full_episode_shape_acc[in_support_sel])
                if len(full_episode_shape_acc[in_support_sel]) > 0
                else np.nan
            ),
            "query_acc": (
                np.mean(full_episode_acc[in_query])
                if len(full_episode_acc[in_query]) > 0
                else np.nan
            ),
            "query_color_acc": (
                np.mean(full_episode_color_acc[in_query])
                if len(full_episode_color_acc[in_query]) > 0
                else np.nan
            ),
            "query_shape_acc": (
                np.mean(full_episode_shape_acc[in_query])
                if len(full_episode_shape_acc[in_query]) > 0
                else np.nan
            ),
        }
        episode_metrics.append(metric)

        # compute samples
        episode_samples[batch_idx]["yq_predict"] = extract(yq_sel, model_predictions)
        episode_samples[batch_idx]["in_support"] = in_support_sel

    return episode_metrics, episode_samples


def compare_arrays(
    prediction: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare two 2D arrays to check for nonzero value matches and shape matches, handling None values in prediction.

    Args:
        prediction (np.ndarray): A 2D array of size (batch_size x max_length) representing the predicted values (may contain None values).
        target (np.ndarray): A 2D array of size (batch_size x max_length) representing the target values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays of size (batch_size):
            - The first array indicates whether the nonzero elements in each row of `prediction` match `target`.
            - The second array indicates whether the shape of the groups in each row of `prediction` matches `target`.
    """
    assert (
        prediction.ndim == 2 and target.ndim == 2
    ), "Both inputs must be 2D arrays"  # batch_size, grid_size*grid_size
    assert prediction.shape == target.shape, "Both inputs must have the same shape"

    color_comparison = []
    shape_comparison = []

    # iterate over batch
    for pred_sample, target_sample in zip(prediction, target):
        if None in pred_sample:
            # If any element in the prediction is None, consider it a mismatch
            color_comparison.append(0)
            shape_comparison.append(0)
            continue

        pred_nonzero = np.unique(pred_sample[pred_sample != 0])
        target_nonzero = np.unique(target_sample[target_sample != 0])

        color_comparison.append(
            1 if np.array_equal(pred_nonzero, target_nonzero) else 0
        )

        predict_shape = {
            coordinates2shape(group)
            for group in find_groups(pattern_to_matrix(pred_sample))
        }
        target_shape = {
            coordinates2shape(group)
            for group in find_groups(pattern_to_matrix(target_sample))
        }
        shape_comparison.append(1 if predict_shape == target_shape else 0)

    return np.array(color_comparison), np.array(shape_comparison)


def compute_accuracy(
    targets: np.ndarray,
    model_preds: np.ndarray,
) -> Dict[str, np.ndarray]:
    # compute exact match accuracy
    """
    Compute exact match accuracy and color, shape accuracy.

    Args:
        targets (np.ndarray): 2D array of shape (batch_size, grid_size*grid_size) of target values.
        model_preds (np.ndarray): 2D array of shape (batch_size, grid_size*grid_size) of model predictions.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the following accuracy metrics:
            - "exact_match_acc": A 1D array of size (batch_size) containing the exact match accuracy for each sample.
            - "color_acc": A 1D array of size (batch_size) containing the color accuracy for each sample.
            - "shape_acc": A 1D array of size (batch_size) containing the shape accuracy for each sample.
    """
    comparison = model_preds == targets
    exact_match_acc = comparison.all(axis=1).astype(int)

    # compute more fine-grained (color and shape) accuracy
    color_accuracies, shape_accuracies = compare_arrays(model_preds, targets)

    return {
        "exact_match_acc": exact_match_acc,
        "color_acc": color_accuracies,
        "shape_acc": shape_accuracies,
    }


def evaluate_acc(
    batch: Dict[str, Any],
    model_logits: torch.Tensor,
    max_length: int,
    tokenizer: Optional[BaseTokenizer] = None,
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of a model on a given batch.

    Args:
        batch (Dict[str, Any]): A dictionary containing the batch data.
        model_logits (torch.Tensor): A 3D tensor of shape (batch_size, sequence_length, vocabulary_size) containing the model predictions.
        max_length (int): The maximum sequence length to consider.
        tokenizer (Optional[BaseTokenizer], optional): An optional tokenizer to use to decode the model predictions. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the mean accuracy metrics across the batch, including:
            - "full_episode_acc": The mean accuracy of the model on the full episode.
            - "full_episode_color_acc": The mean color accuracy of the model on the full episode.
            - "full_episode_shape_acc": The mean shape accuracy of the model on the full episode.
            - "copy_acc": The mean accuracy of the model on the support set.
            - "copy_color_acc": The mean color accuracy of the model on the support set.
            - "copy_shape_acc": The mean shape accuracy of the model on the support set.
            - "query_acc": The mean accuracy of the model on the query set.
            - "query_color_acc": The mean color accuracy of the model on the query set.
            - "query_shape_acc": The mean shape accuracy of the model on the query set.
    """
    # greedy decoding
    max_logits = torch.argmax(model_logits, dim=2)[:, :max_length]

    # decode predictions if tokenizer is used
    if tokenizer:
        model_preds = [tokenizer.decode(pred.tolist()) for pred in max_logits]
        model_preds = np.array(flatten_matrices(model_preds))
    else:
        model_preds = max_logits.cpu().numpy()

    # get target
    targets = np.array(flatten_matrices(batch["yq"]))

    # compute accuracies
    acc_dict = compute_accuracy(
        targets=targets,
        model_preds=model_preds,
    )
    episode_metrics, episode_samples = extract_episode_accuracy(
        list_samples=batch["list_samples"],
        in_support=np.array(batch["in_support"]),
        model_predictions=model_preds,
        acc_dict=acc_dict,
        q_idx=batch["q_idx"].cpu().numpy(),
    )

    # compute and return mean accuracies across the batch
    return {
        "samples_pred": episode_samples,
        "full_episode_acc": np.mean(
            [sample["full_episode_acc"] for sample in episode_metrics]
        ),
        "full_episode_color_acc": np.mean(
            [sample["full_episode_color_acc"] for sample in episode_metrics]
        ),
        "full_episode_shape_acc": np.mean(
            [sample["full_episode_shape_acc"] for sample in episode_metrics]
        ),
        "copy_acc": np.mean([sample["copy_acc"] for sample in episode_metrics]),
        "copy_color_acc": np.mean(
            [sample["copy_color_acc"] for sample in episode_metrics]
        ),
        "copy_shape_acc": np.mean(
            [sample["copy_shape_acc"] for sample in episode_metrics]
        ),
        "query_acc": np.mean([sample["query_acc"] for sample in episode_metrics]),
        "query_color_acc": np.mean(
            [sample["query_color_acc"] for sample in episode_metrics]
        ),
        "query_shape_acc": np.mean(
            [sample["query_shape_acc"] for sample in episode_metrics]
        ),
    }


def smooth_decoder_outputs(
    logits_flat: torch.Tensor,
    p_lapse: float,
    lapse_symb_include: List[str],
    langs: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Mix decoder outputs (logits_flat) with a uniform distribution over allowed emissions.

    Args:
        logits_flat (torch.Tensor): Logits from the decoder (batch*max_len, output_size).
        p_lapse (float): Probability of a uniform lapse.
        lapse_symb_include (List[str]): List of tokens to include in the lapse model.
        langs (Dict[str, Any]): Dictionary of Lang classes.
        device (torch.device): Device to run the computation on.

    Returns:
        torch.Tensor: Normalized log-probabilities.
    """
    lapse_idx_include = [langs["output"].symbol2index[s] for s in lapse_symb_include]
    sz = logits_flat.size()  # get size (batch*max_len, output_size)
    probs_flat = F.softmax(logits_flat, dim=1)  # (batch*max_len, output_size)
    num_classes_lapse = len(lapse_idx_include)
    probs_lapse = torch.zeros(sz, dtype=torch.float)
    probs_lapse = probs_lapse.to(device)
    probs_lapse[:, lapse_idx_include] = 1.0 / float(num_classes_lapse)
    log_probs_flat = torch.log(
        (1 - p_lapse) * probs_flat + p_lapse * probs_lapse
    )  # (batch*max_len, output_size)
    return log_probs_flat


def compare_lists(list1: List[Any], list2: List[Any]) -> List[int]:
    """
    Compare two lists element-wise and return a list indicating matches.

    Args:
        list1 (List[Any]): The first list to compare.
        list2 (List[Any]): The second list to compare.

    Returns:
        List[int]: A list of integers where each element is 1 if the corresponding elements in `list1` and `list2` are equal, and 0 otherwise.

    Raises:
        ValueError: If the two lists do not have the same length.
    """
    # Ensure both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")

    # Perform element-wise comparison
    comparison = [1 if list1[i] == list2[i] else 0 for i in range(len(list1))]

    return comparison
