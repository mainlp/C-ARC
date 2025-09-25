"""
Code modified based on:
Lake, B. M. and Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623, 115-121.
<https://github.com/brendenlake/MLC/blob/main/datasets.py>

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
from typing import Any, Dict, Literal

from torch.utils.data import Dataset

from vmlc.utils.utils import add_response_noise, load_jsonl, parse_data


class VisualGrammarDataset(Dataset):
    """
    Meta-training for few-shot grammar learning
    """

    def __init__(
        self,
        data_dir: str,
        data_file_name: str,
        mode: Literal["train", "val", "test"],
        p_noise: float = 0.0,
        test_queries: int = -1,
        with_copy_task: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Each episode has a different latent (algebraic) grammar.
        The number of support items is picked uniformly from min_ns to max_ns.

        Args:
            data_dir (str): Directory where the data is stored.
            data_file_name (str): Filename of data file.
            mode (Literal["train", "val", "test"]): 'train', 'val' or 'test' indicating the mode of the dataset.
            p_noise (float, optional): Probability that a given symbol emission will be from a uniform distribution. Defaults to 0.1.
            with_copy_task (bool, optional): Whether support items should also be query items. Defaults to True.
        """
        self.mode = mode
        self.train = mode == "train"
        self.p_noise = p_noise
        self.test_queries = test_queries
        self.with_copy_task = with_copy_task

        self.data_samples = load_jsonl(
            file_path=os.path.join(data_dir, f"{mode}_{data_file_name}.jsonl")
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an episode from the dataset.

        Args:
            idx (int): Index of the episode to retrieve.

        Returns:
            Dict[str, Any]: A dictionary representing the episode.
        """
        sample = parse_data(episode=self.data_samples[idx])

        # subsample number of test queries if specified
        if self.test_queries > 0:
            sample["xq"] = sample["xq"][: self.test_queries]
            sample["yq"] = sample["yq"][: self.test_queries]

        # randomly add noise for decoder training robustness
        if self.train and self.p_noise > 0:
            for i in range(len(sample["yq"])):
                sample["yq"][i] = add_response_noise(sample["yq"][i], self.p_noise)

        if self.with_copy_task:
            sample["xq"] = sample["xs"] + sample["xq"]
            sample["yq"] = sample["ys"] + sample["yq"]

        return sample

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data_samples)
