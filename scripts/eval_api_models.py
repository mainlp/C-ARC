"""
Evaluate the batch output of API models model (e.g. OpenAI's gpt-4o-2024-08-06).
"""

import argparse
import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from vmlc.model.metrics import compute_accuracy
from vmlc.utils import load_jsonl, save_dict_to_json, set_seed, setup_logging
from vmlc.utils.plot_utils import display_input_output_plot


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Evaluate API model batch output.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=1860, help="Random generator seed.")
    parser.add_argument(
        "--batch_data_dir",
        type=str,
        required=True,
        help="The directory to where the batch output is stored.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="The file path to the original test data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory output files should be stored to.",
    )

    # Plot configs
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=0,
        help="The frequency with which samples are plotted.",
    )

    return parser.parse_args()


def convert_to_2d_array_string(input_str: str) -> np.ndarray:
    """
    Extracts a 10x10 2D array from an input string, cleans extraneous text
    such as markdown code fences, language markers, and 'output =' labels,
    and returns a NumPy array representation of the array.

    Args:
        input_str (str): The input string containing extraneous text around a 2D array.

    Returns:
        np.ndarray: A NumPy array representing the 10x10 2D array.

    Raises:
        ValueError: If no valid 2D array can be extracted or if the extracted array is not 10x10.
    """
    input_str = input_str.lower()

    if "output" not in input_str:
        raise ValueError(f"{input_str}\nNo 'output' marker found in prediction!")

    # Initial cleaning: strip whitespace
    cleaned = input_str.strip()

    # Remove markdown code fences and any language specifier (like "python")
    cleaned = re.sub(r"^```(?!\s*output\b)[\w]*\n", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("```", "")

    # Use regex to locate the "output" marker and get the substring after it.
    output_pattern = re.compile(r"output", re.IGNORECASE)
    match = output_pattern.search(cleaned)
    if not match:
        raise ValueError(
            f"{input_str}\nNo 'output' marker found in the cleaned input: {cleaned}!"
        )

    # Only work with the text after the "output" marker.
    after_output = cleaned[match.end() :]

    # Find the first '[' after the cleaned keyword or from the beginning
    bracket_index = after_output.find("[")
    if bracket_index == -1:
        raise ValueError(f"{input_str}\nNo array found in the input string.")

    def extract_brackets(s: str, start: int) -> str | None:
        count = 0
        start_index = None
        for i, char in enumerate(s[start:], start=start):
            if char == "[":
                if count == 0:
                    start_index = i
                count += 1
            elif char == "]":
                count -= 1
                if count == 0:
                    return s[start_index : i + 1]
        return None

    array_str = extract_brackets(after_output, bracket_index)
    if array_str is None:
        raise ValueError(
            f"{input_str}\nCould not extract a valid array from the input string: {after_output}"
        )

    # Convert the extracted string into a Python object using literal_eval
    try:
        array_obj = ast.literal_eval(array_str)
    except Exception as e:
        raise ValueError(
            f"{input_str}\nExtracted array string could not be parsed as a valid Python object from: {array_str}"
        ) from e

    # Validate that the array is 10x10 (a list of 10 lists each of length 10)
    if not isinstance(array_obj, list) or len(array_obj) != 10:
        raise ValueError(
            f"{input_str}\nArray must be a list of 10 rows; found {len(array_obj)} rows in {array_obj}."
        )
    for i, row in enumerate(array_obj):
        if not isinstance(row, list):
            raise ValueError(f"{input_str}\nRow {row} not a list in {array_obj}!.")
        if len(row) != 10:
            raise ValueError(
                f"{input_str}\nRow {i} must be a list of 10 elements; found {len(row)} elements in {array_obj}."
            )

    # Convert the list of lists into a NumPy array and check its shape
    array_np = np.array(array_obj)
    if array_np.shape != (10, 10):
        raise ValueError(
            f"{input_str}\nExtracted array shape is {array_np.shape} but expected (10, 10)."
        )
    return array_np


def extract_batched_response_gpt(
    batched_data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
    """
    Extract responses from batch.

    Args:
        batched_data (List[Dict[str, Any]]): The batch of response data.

    Returns:
        List[Dict[str, np.ndarray]]: The model response.
    """
    valid_responses: int = 0
    invalid_responses: int = 0
    filtered_batched_data: List[Dict[str, np.ndarray]] = []
    for data_dict in batched_data:
        if data_dict["error"] is None:
            custom_id = data_dict["custom_id"]
            evaluator_response = data_dict["evaluator_response"]
            try:
                array = convert_to_2d_array_string(evaluator_response)
                valid_responses += 1
            except ValueError as e:
                logging.error(e)
                invalid_responses += 1
                continue
            filtered_batched_data.append({custom_id: array})
        else:
            invalid_responses += 1

    return filtered_batched_data, valid_responses, invalid_responses


def extract_batched_response_gemini(
    batched_data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
    """
    Extract responses from batch.

    Args:
        batched_data (List[Dict[str, Any]]): The batch of response data.

    Returns:
        List[Dict[str, np.ndarray]]: The model response.
    """
    valid_responses: int = 0
    invalid_responses: int = 0
    filtered_batched_data: List[Dict[str, np.ndarray]] = []
    for data_dict in batched_data:
        if data_dict["status"] == "":
            custom_id = data_dict["request"]["labels"]["custom_id"]
            evaluator_response = data_dict["response"]["candidates"][0]["content"][
                "parts"
            ][0]["text"]
            try:
                array = convert_to_2d_array_string(evaluator_response)
                valid_responses += 1
            except ValueError as e:
                print(e)
                print(50 * "====")
                invalid_responses += 1
                continue
            filtered_batched_data.append({custom_id: array})

    return filtered_batched_data, valid_responses, invalid_responses


def load_batched_data(
    batch_data_dir: str, model_type: Literal["gpt", "gemini"]
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
    """
    Loads and processes batched response data from JSONL files in a directory.

    Args:
        batch_data_dir (str): The directory containing the JSONL files with
                              batched response data.
        model_type (Literal["gpt", "gemini"]): Type of API model to consider.

    Returns:
        List[Dict[str, np.ndarray]]: A list of dictionaries where each dictionary
                                     contains a custom ID and its corresponding
                                     2D NumPy array extracted from the evaluator
                                     response.
    """
    batched_responses: List[Dict[str, np.ndarray]] = []
    valid_responses: int = 0
    invalid_responses: int = 0

    for jsonl_file in Path(batch_data_dir).glob("*.jsonl"):
        response_data = load_jsonl(str(jsonl_file))
        if model_type == "gpt":
            filtered_response_dict, valid_answers, invalid_answers = (
                extract_batched_response_gpt(response_data)
            )
        elif model_type == "gemini":
            filtered_response_dict, valid_answers, invalid_answers = (
                extract_batched_response_gemini(response_data)
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        valid_responses += valid_answers
        invalid_responses += invalid_answers
        batched_responses.extend(filtered_response_dict)

    return batched_responses, valid_responses, invalid_responses


def compute_test_acc(
    predictions: List[Dict[str, np.ndarray]],
    target_data: List[Dict[str, Any]],
    output_dir: str,
    plot_freq: int = 0,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes the test accuracy from batched predictions and target data.

    Args:
        predictions (List[Dict[str, np.ndarray]]): A list of dictionaries where each
            dictionary contains a custom ID and its corresponding 2D NumPy array
            extracted from the evaluator response.
        target_data (List[Dict[str, Any]]): A list of dictionaries containing the target
            data.

    Returns:
        Dict[str, Tuple[float, float]]: A dictionary containing the mean and standard
            deviation of the query accuracy, color accuracy, and shape accuracy.
    """
    pred_arrays_reshaped_list: List[np.ndarray] = []
    target_arrays_reshaped_list: List[np.ndarray] = []

    num_samples = 0

    for prediction_dict in predictions:
        sample_nums = list(prediction_dict.keys())
        assert len(sample_nums) == 1, f"More than one key in dict! {prediction_dict}"

        sample_num = sample_nums[0]
        pred_array = prediction_dict[sample_num]

        indicees = re.findall(r"\d+", sample_num)
        assert len(indicees) == 1, f"Invalid number of digits in string! {sample_num}"
        idx = int(indicees[0])

        # get target array
        target_array = np.array(target_data[idx]["queries"][0][1])

        if plot_freq > 0 and num_samples % plot_freq == 0:
            study_examples: List[np.ndarray] = []
            for input_output_pair in target_data[idx]["study_examples"]:
                study_examples += [
                    np.array(input_output_pair[0]),
                    np.array(input_output_pair[1]),
                ]

            input_array = np.array(target_data[idx]["queries"][0][0])
            display_input_output_plot(
                patterns=[study_examples + [input_array, pred_array, target_array]],
                plot_name=f"{output_dir}/plots/sample_{idx}.png",
            )

        # reshape for acc computation
        target_array_reshaped = np.array(target_array).reshape(1, -1)
        pred_array_reshaped = pred_array.reshape(1, -1)

        target_arrays_reshaped_list.append(target_array_reshaped)
        pred_arrays_reshaped_list.append(pred_array_reshaped)

        num_samples += 1

    acc_dict = compute_accuracy(
        targets=np.vstack(target_arrays_reshaped_list),
        model_preds=np.vstack(pred_arrays_reshaped_list),
    )

    return {label: np.mean(output) for label, output in acc_dict.items()}


def fetch_model_type(batch_data_dir: str) -> str:
    """
    Infers the model type based on the directory name.

    Args:
        batch_data_dir (str): The directory name containing the batch data.

    Returns:
        str: The model type as a string, either "gemini" or "gpt".

    Raises:
        ValueError: If the model type cannot be inferred from the directory name.
    """
    if "gemini" in batch_data_dir:
        return "gemini"
    elif "gpt-4o" in batch_data_dir:
        return "gpt"
    elif "o3-mini" in batch_data_dir:
        return "gpt"
    else:
        raise ValueError(f"Unable to infer model type from: {batch_data_dir}")


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # get batch data
    model_type = fetch_model_type(args.batch_data_dir)
    batched_responses, valid_responses, invalid_responses = load_batched_data(
        batch_data_dir=args.batch_data_dir, model_type=model_type  # type: ignore
    )

    # get original test data
    test_data = load_jsonl(file_path=args.test_data_path)

    # compute accuracy
    acc = compute_test_acc(
        predictions=batched_responses,
        target_data=test_data,
        output_dir=args.output_dir,
        plot_freq=args.plot_freq,
    )
    acc["valid responses"] = valid_responses  # type: ignore
    acc["invalid responses"] = invalid_responses  # type: ignore
    logging.info(f"Accuracy:\n{acc}")

    save_dict_to_json(data=acc, file_path=f"{args.output_dir}/model_accuracy.json")


if __name__ == "__main__":
    main()
