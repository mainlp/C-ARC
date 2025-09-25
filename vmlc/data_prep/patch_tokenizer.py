import itertools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from vmlc.data_prep.base_tokenizer import BaseTokenizer
from vmlc.data_prep.constants import EOS_TOKEN, IO_SEP, PAD_TOKEN, SOS_TOKEN, SUP_SEP


class PatchTokenizer(BaseTokenizer):
    def __init__(self, patch_size: int = 2, max_value: int = 9, grid_size=10):
        """
        Initializes the PatchTokenizer with a specified patch size and range of values.

        Args:
            patch_size (int): The size of the square patches (default is 2 for 2x2 patches).
            max_value (int): Maximum value for each pixel (e.g., 9 for values 0-9).
        """
        self.patch_size = patch_size
        self.max_value = max_value
        self.grid_size = grid_size

        # Initialize vocab
        self.token_to_id, self.id_to_token = {}, {}
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, IO_SEP, SUP_SEP]
        self.zero_token = (0, 0, 0, 0)

        # Add special tokens to vocab
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        self._build_vocab()

        self.PAD_idx = self.token_to_id[PAD_TOKEN]
        self.vocab_size = len(self.token_to_id)

    def _build_vocab(
        self,
    ) -> None:
        """
        Builds a vocabulary mapping patches to unique token IDs and vice versa.
        Special tokens occupy the first positions in the vocabulary.

        This function updates the following attributes:
        - `self.token_to_id`: Mapping from patch tuple to token ID.
        - `self.id_to_token`: Mapping from token ID to patch tuple.
        """

        # Enumerate all possible patch configurations
        possible_values = range(self.max_value + 1)
        num_elements_per_patch = self.patch_size * self.patch_size
        patch_combinations = itertools.product(
            possible_values, repeat=num_elements_per_patch
        )

        # First positions are reserved for special tokens
        start_idx = len(self.id_to_token)
        for idx, patch in enumerate(patch_combinations, start=start_idx):
            if patch == self.zero_token:
                self.zero_token_id = idx
            self.token_to_id[patch] = idx
            self.id_to_token[idx] = patch

    def _tokenize_single_grid(self, grid: np.ndarray) -> List[int]:
        """
        Converts a single 2D grid into a sequence of token IDs representing patches.

        Args:
            grid (np.ndarray): 2D grid (e.g., 10x10) of integers.

        Returns:
            List[int]: Sequence of token IDs for the patches.
        """
        tokens = []

        # Slide through the grid in non-overlapping patch_size x patch_size steps
        for i in range(0, grid.shape[0], self.patch_size):
            for j in range(0, grid.shape[1], self.patch_size):
                patch = tuple(
                    grid[i : i + self.patch_size, j : j + self.patch_size].flatten()
                )
                tokens.append(self.token_to_id[patch])

        return tokens

    def tokenize(
        self, grids: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Converts a list of tuples of 2D grids into a list of tuples of tokenized grids.

        Args:
            grids (List[Tuple[np.ndarray, np.ndarray]]):
                List of tuples, where each tuple contains two 2D grids.

        Returns:
            List[Tuple[List[int], List[int]]]:
                List of tuples of tokenized grids (lists of token IDs).
        """
        tokenized_grids = []

        for grid1, grid2 in grids:
            assert isinstance(
                grid1, np.ndarray
            ), f"Invalid type of grid1. Should be np.ndarray: {grid1}"
            assert isinstance(
                grid2, np.ndarray
            ), f"Invalid type of grid1. Should be np.ndarray: {grid2}"

            # Tokenize each grid in the tuple
            tokens1 = self._tokenize_single_grid(grid1)
            tokens2 = self._tokenize_single_grid(grid2)

            # Append the tokenized grids as a tuple
            tokenized_grids.append((tokens1, tokens2))

        return tokenized_grids

    def decode(self, tokens: Sequence[int]) -> List[List[Optional[int]]]:
        """
        Converts a sequence of token IDs back into the original 2D grid.

        Args:
            tokens (Sequence[int]): Sequence of token IDs representing patches.

        Returns:
            List[List[Optional[int]]]: Reconstructed 2D grid.
        """
        grid: List[List[Optional[int]]] = [
            [None for _ in range(self.grid_size)] for _ in range(self.grid_size)
        ]
        token_idx: int = 0

        # Reconstruct the grid row by row
        for i in range(0, self.grid_size, self.patch_size):
            for j in range(0, self.grid_size, self.patch_size):
                # patch can either be a special token (an int) or a list of ints representing the patch.
                patch: Union[int, List[int]] = self.id_to_token[tokens[token_idx]]

                if patch == EOS_TOKEN:
                    break
                elif patch in self.special_tokens:
                    token_idx += 1
                    continue
                else:
                    # Here, patch is assumed to be a list of ints.
                    reshaped_patch: List[List[int]] = [
                        patch[row * self.patch_size : (row + 1) * self.patch_size]  # type: ignore
                        for row in range(self.patch_size)
                    ]

                # Insert the reshaped patch into the grid
                for patch_row_idx, patch_row in enumerate(reshaped_patch):
                    grid[i + patch_row_idx][j : j + self.patch_size] = patch_row

                token_idx += 1

        return grid
