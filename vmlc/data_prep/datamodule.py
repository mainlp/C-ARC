from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from vmlc.data_prep.base_tokenizer import BaseTokenizer
from vmlc.data_prep.dataset import VisualGrammarDataset


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_file_name: str,
        tokenizer: BaseTokenizer,
        batch_size: int = 32,
        p_noise: float = 0.01,
        with_copy_task: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        """
        DataModule for the Compositional-ARC dataset.

        Args:
            data_dir (str): Directory for the data.
            data_file_name (str): Filename of data file.
            tokenizer (BaseTokenizer): Tokenizer for grid encoding and decoding.
            batch_size (int, optional): Batch size for training and validation. Defaults to 32.
            p_noise (float, optional): Noise probability for the training dataset. Defaults to 0.01.
            with_copy_task (bool, optional): Include support items in the query for copy task. Defaults to True.
            num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
            prefetch_factor (int, optional): Prefetch factor for DataLoader. Defaults to 2.
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_file_name = data_file_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.p_noise = p_noise
        self.with_copy_task = with_copy_task
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = VisualGrammarDataset(
                data_dir=self.data_dir,
                data_file_name=self.data_file_name,
                mode="train",
                p_noise=self.p_noise,
                with_copy_task=self.with_copy_task,
            )
            self.val_dataset = VisualGrammarDataset(
                data_dir=self.data_dir,
                data_file_name=self.data_file_name,
                mode="val",
                p_noise=0.0,
                with_copy_task=self.with_copy_task,
            )
        if stage == "test":
            self.test_dataset = VisualGrammarDataset(
                data_dir=self.data_dir,
                data_file_name=self.data_file_name,
                mode="test",
                p_noise=0.0,
                with_copy_task=self.with_copy_task,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=lambda x: self.make_batch_from_episodes(x),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=lambda x: self.make_batch_from_episodes(x),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=lambda x: self.make_batch_from_episodes(x),
        )

    def _batch_tokenized_sequences(
        self,
        tokenized_samples: List[Dict[str, List[Tuple[List[int], List[int]]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create batched input sequences and target sequences with SOS and EOS tokens.
        Additionally, compute grid boundaries for each flattened sequence.

        Args:
            tokenized_samples (List[Dict]): List of tokenized samples with "support_io" and "query_io".

        Returns:
            Tuple containing:
                - final_xq_context (torch.Tensor): Batched input sequences.
                - final_targets_sos (torch.Tensor): Targets prepended with SOS.
                - final_targets_eos (torch.Tensor): Targets appended with EOS.
        """
        batched_xq_context = []
        batched_targets_sos = []
        batched_targets_eos = []

        for batch_element in tokenized_samples:
            for query_input, query_output in batch_element["query_io"]:
                xq_context = []

                # Add all support_io pairs to xq_context
                for input_support, output_support in batch_element["support_io"]:
                    xq_context.append(input_support)
                    xq_context.append([self.tokenizer.token_to_id["IO_SEP"]])
                    xq_context.append(output_support)
                    xq_context.append([self.tokenizer.token_to_id["SUP_SEP"]])

                # Add the query to xq_context
                xq_context.append(query_input + [self.tokenizer.token_to_id["IO_SEP"]])

                flattened_sequence = [
                    item for sublist in xq_context for item in sublist
                ]
                batched_xq_context.append(
                    torch.tensor(flattened_sequence, dtype=torch.int64)
                )

                # Add SOS and EOS to targets
                batched_targets_sos.append(
                    torch.tensor(
                        [self.tokenizer.token_to_id["SOS"]] + query_output,
                        dtype=torch.int64,
                    )
                )
                batched_targets_eos.append(
                    torch.tensor(
                        query_output + [self.tokenizer.token_to_id["EOS"]],
                        dtype=torch.int64,
                    )
                )

        final_xq_context = torch.stack(batched_xq_context)
        final_targets_sos = torch.stack(batched_targets_sos)
        final_targets_eos = torch.stack(batched_targets_eos)

        return final_xq_context, final_targets_sos, final_targets_eos

    def _tokenize_episode(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, List[Tuple[List[int], List[int]]]]:
        """
        Tokenizes a single episode into support and query IO pairs.

        Args:
            sample (Dict[str, Any]): A single episode containing 'xs', 'ys', 'xq', and 'yq'.

        Returns:
            Dict[str, List[Tuple[List[int], List[int]]]]: Tokenized support and query pairs.
        """
        support_io: List[Tuple[np.ndarray, np.ndarray]] = []
        query_io: List[Tuple[np.ndarray, np.ndarray]] = []

        for xs, ys in zip(sample["xs"], sample["ys"]):
            support_io.append((xs, ys))

        for xq, yq in zip(sample["xq"], sample["yq"]):
            query_io.append((xq, yq))

        return {
            "support_io": self.tokenizer.tokenize(support_io),
            "query_io": self.tokenizer.tokenize(query_io),
        }

    def make_batch_from_episodes(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Converts episodic data samples into a batched format for model training.

        Args:
            samples (List[Dict[str, Any]]): List of episodes, where each episode contains 'xs', 'ys', 'xq', 'yq'.

        Returns:
            Dict[str, Any]: Dictionary containing batched tensors and related information.
        """
        # Initialize outputs
        q_idx_list: List[torch.Tensor] = []
        in_support_list: List[bool] = []
        yq_list: List[np.ndarray] = []
        tokenized_samples: List[Dict[str, List[Tuple[List[int], List[int]]]]] = []

        # Tokenize episodes
        for idx, sample in enumerate(samples):
            tokenized_sample = self._tokenize_episode(sample)
            tokenized_samples.append(tokenized_sample)

            # Extract query indices, in_support flags, and raw targets
            nq = len(sample["xq"])
            q_idx_list.append(torch.full((nq,), idx, dtype=torch.int))
            for xq_sample in sample["xq"]:
                is_in_support = any(
                    np.array_equal(xq_sample, arr) for arr in sample["xs"]
                )
                in_support_list.append(is_in_support)
            yq_list.extend(sample["yq"])

        q_idx = torch.cat(q_idx_list)

        # Create batched sequences
        xq_context, yq_sos, yq_eos = self._batch_tokenized_sequences(tokenized_samples)

        return {
            "list_samples": samples,
            "q_idx": q_idx,
            "in_support": in_support_list,
            "yq": yq_list,
            "xq_context": xq_context,
            "yq_sos": yq_sos,
            "yq_eos": yq_eos,
        }


class LitTestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_file_name: str,
        tokenizer: BaseTokenizer,
        batch_size: int = 32,
        p_noise: float = 0.01,
        with_copy_task: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        test_queries: int = -1,
    ):
        """
        TestDataModule for the Compositional-ARC dataset.

        Args:
            data_dir (str): Directory for the data.
            data_file_name (str): Filename of data file.
            tokenizer (BaseTokenizer): Tokenizer for grid encoding and decoding.
            batch_size (int, optional): Batch size for training and validation. Defaults to 32.
            p_noise (float, optional): Noise probability for the training dataset. Defaults to 0.01.
            with_copy_task (bool, optional): Include support items in the query for copy task. Defaults to True.
            num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
            prefetch_factor (int, optional): Prefetch factor for DataLoader. Defaults to 2.
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_file_name = data_file_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.p_noise = p_noise
        self.with_copy_task = with_copy_task
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.test_queries = test_queries

    def setup(self, stage: Optional[str] = None):
        self.test_dataset = VisualGrammarDataset(
            data_dir=self.data_dir,
            data_file_name=self.data_file_name,
            mode="test",
            p_noise=0.0,
            test_queries=self.test_queries,
            with_copy_task=self.with_copy_task,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=lambda x: self.make_batch_from_episodes(x),
        )

    def _batch_tokenized_sequences(
        self,
        tokenized_samples: List[Dict[str, List[Tuple[List[int], List[int]]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create batched input sequences and target sequences with SOS and EOS tokens.
        Additionally, compute grid boundaries for each flattened sequence.

        Args:
            tokenized_samples (List[Dict]): List of tokenized samples with "support_io" and "query_io".

        Returns:
            Tuple containing:
                - final_xq_context (torch.Tensor): Batched input sequences.
                - final_targets_sos (torch.Tensor): Targets prepended with SOS.
                - final_targets_eos (torch.Tensor): Targets appended with EOS.
        """
        batched_xq_context = []
        batched_targets_sos = []
        batched_targets_eos = []

        for batch_element in tokenized_samples:
            for query_input, query_output in batch_element["query_io"]:
                xq_context = []

                # Add all support_io pairs to xq_context
                for input_support, output_support in batch_element["support_io"]:
                    xq_context.append(input_support)
                    xq_context.append([self.tokenizer.token_to_id["IO_SEP"]])
                    xq_context.append(output_support)
                    xq_context.append([self.tokenizer.token_to_id["SUP_SEP"]])

                # Add the query to xq_context
                xq_context.append(query_input + [self.tokenizer.token_to_id["IO_SEP"]])

                flattened_sequence = [
                    item for sublist in xq_context for item in sublist
                ]
                batched_xq_context.append(
                    torch.tensor(flattened_sequence, dtype=torch.int64)
                )

                # Add SOS and EOS to targets
                batched_targets_sos.append(
                    torch.tensor(
                        [self.tokenizer.token_to_id["SOS"]] + query_output,
                        dtype=torch.int64,
                    )
                )
                batched_targets_eos.append(
                    torch.tensor(
                        query_output + [self.tokenizer.token_to_id["EOS"]],
                        dtype=torch.int64,
                    )
                )

        final_xq_context = torch.stack(batched_xq_context)
        final_targets_sos = torch.stack(batched_targets_sos)
        final_targets_eos = torch.stack(batched_targets_eos)

        return final_xq_context, final_targets_sos, final_targets_eos

    def _tokenize_episode(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, List[Tuple[List[int], List[int]]]]:
        """
        Tokenizes a single episode into support and query IO pairs.

        Args:
            sample (Dict[str, Any]): A single episode containing 'xs', 'ys', 'xq', and 'yq'.

        Returns:
            Dict[str, List[Tuple[List[int], List[int]]]]: Tokenized support and query pairs.
        """
        support_io: List[Tuple[np.ndarray, np.ndarray]] = []
        query_io: List[Tuple[np.ndarray, np.ndarray]] = []

        for xs, ys in zip(sample["xs"], sample["ys"]):
            support_io.append((xs, ys))

        for xq, yq in zip(sample["xq"], sample["yq"]):
            query_io.append((xq, yq))

        return {
            "support_io": self.tokenizer.tokenize(support_io),
            "query_io": self.tokenizer.tokenize(query_io),
        }

    def make_batch_from_episodes(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Converts episodic data samples into a batched format for model training.

        Args:
            samples (List[Dict[str, Any]]): List of episodes, where each episode contains 'xs', 'ys', 'xq', 'yq'.

        Returns:
            Dict[str, Any]: Dictionary containing batched tensors and related information.
        """
        # Initialize outputs
        q_idx_list: List[torch.Tensor] = []
        in_support_list: List[bool] = []
        yq_list: List[np.ndarray] = []
        tokenized_samples: List[Dict[str, List[Tuple[List[int], List[int]]]]] = []

        # Tokenize episodes
        for idx, sample in enumerate(samples):
            tokenized_sample = self._tokenize_episode(sample)
            tokenized_samples.append(tokenized_sample)

            # Extract query indices, in_support flags, and raw targets
            nq = len(sample["xq"])
            q_idx_list.append(torch.full((nq,), idx, dtype=torch.int))
            for xq_sample in sample["xq"]:
                is_in_support = any(
                    np.array_equal(xq_sample, arr) for arr in sample["xs"]
                )
                in_support_list.append(is_in_support)
            yq_list.extend(sample["yq"])

        q_idx = torch.cat(q_idx_list)

        # Create batched sequences
        xq_context, yq_sos, yq_eos = self._batch_tokenized_sequences(tokenized_samples)

        return {
            "list_samples": samples,
            "q_idx": q_idx,
            "in_support": in_support_list,
            "yq": yq_list,
            "xq_context": xq_context,
            "yq_sos": yq_sos,
            "yq_eos": yq_eos,
        }
