"""
Modules and functions for positional encodings.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn


class SinCosPositionalEncoding(nn.Module):
    """
    Standard sinusoidal 1D positional encoding.
    """

    def __init__(self, max_seq_len: int, emb_size: int, dropout: float = 0.1):
        super(SinCosPositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
        pos = torch.arange(0, max_seq_len).reshape(max_seq_len, 1)

        pos_embedding = torch.zeros((max_seq_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # shape: [1, maxlen, emb_size]

        self.register_buffer("pos_embedding", pos_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, emb_size].

        Returns:
            torch.Tensor: Tensor with standard 1D positional embeddings added.
        """
        return self.dropout(x + self.pos_embedding[:, : x.size(1), :])


class LearnedPositionalEncoding1D(nn.Module):
    """
    Learned 1D positional encoding.

    Attributes:
        pos_embedding (nn.Parameter): A learnable parameter of shape [1, max_seq_len, emb_size].
        dropout (nn.Dropout): Dropout applied after adding the positional encoding.
    """

    def __init__(self, max_seq_len: int, emb_size: int, dropout: float = 0.1) -> None:
        super(LearnedPositionalEncoding1D, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, emb_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, emb_size].

        Returns:
            torch.Tensor: Input tensor with positional embeddings added.
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pos_embedding[:, :seq_len, :])


class LearnedPositionalEncoding2D(nn.Module):
    """
    Implements 2D positional encoding by adding separate learnable row and column embeddings.
    For a 10x10 grid encoded with 2x2 patches, the grid becomes 5x5 tokens.
    """

    def __init__(
        self,
        num_patches_x: int,
        num_patches_y: int,
        emb_size: int,
        dropout: float = 0.1,
    ):
        super(LearnedPositionalEncoding2D, self).__init__()
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.emb_size = emb_size

        # Learnable embeddings for rows and columns.
        self.row_embed = nn.Parameter(torch.randn(num_patches_x, emb_size))
        self.col_embed = nn.Parameter(torch.randn(num_patches_y, emb_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token embeddings of shape [batch_size, seq_len, emb_size].
                              Here seq_len should equal grid_height * grid_width.
        Returns:
            torch.Tensor: x with added 2D positional embeddings.
        """
        _, seq_len, _ = x.shape
        assert (
            seq_len == self.num_patches_x * self.num_patches_y
        ), f"Expected sequence length {self.num_patches_x * self.num_patches_y} but got {seq_len}"

        # Compute row and column indices for each token.
        device = x.device
        rows = (
            torch.arange(self.num_patches_x, device=device)
            .unsqueeze(1)
            .repeat(1, self.num_patches_y)
            .flatten()
        )
        cols = (
            torch.arange(self.num_patches_y, device=device)
            .unsqueeze(0)
            .repeat(self.num_patches_x, 1)
            .flatten()
        )

        # Sum the corresponding row and column embeddings.
        pos_emb = self.row_embed[rows] + self.col_embed[cols]
        pos_emb = pos_emb.unsqueeze(0)

        return self.dropout(x + pos_emb)


class LearnedPositionalEncodingCombined(nn.Module):
    """
    Combined positional encoding that adds a base 1D encoding and injects
    2D grid offsets at positions corresponding to grid tokens.

    The full (maximum) sequence is assumed to follow this structure:
      For each support pair:
        - Input grid block: grid_block_size tokens
        - IO_SEP token: 1 token (separator)
        - Output grid block: grid_block_size tokens
        - SUP_SEP token: 1 token (separator)
      Followed by the query:
        - Query grid block: grid_block_size tokens
        - IO_SEP token: 1 token (separator)

    The maximum sequence length (max_seq_len) is used to precompute grid boundaries.
    At runtime, the 2D encoding is computed on the fly so that gradients flow back
    through the learnable row and column embeddings.
    """

    def __init__(
        self,
        emb_size: int,
        dropout: float = 0.1,
        num_patches_x: int = 5,
        num_patches_y: int = 5,
        num_support: int = 18,
    ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.grid_block_size = num_patches_x * num_patches_y
        self.num_support = num_support
        self.max_seq_len = (
            num_support * (2 * self.grid_block_size + 2) + self.grid_block_size + 1
        )

        # Base 1D positional encoding over the maximum sequence length.
        self.positional_1d = LearnedPositionalEncoding1D(
            self.max_seq_len, emb_size, dropout
        )

        # Learnable 2D embeddings for rows and columns.
        self.row_embed = nn.Parameter(torch.randn(num_patches_x, emb_size))
        self.col_embed = nn.Parameter(torch.randn(num_patches_y, emb_size))
        self.dropout = nn.Dropout(dropout)

        # Compute grid boundaries based on the expected full pattern.
        self.grid_boundaries: List[Tuple[int, int]] = self._compute_grid_boundaries()

        # For target sequences, expected target_seq_len = 1 (SOS) + grid_block_size.
        self.target_seq_len = 1 + self.grid_block_size

    def _compute_grid_boundaries(self) -> List[Tuple[int, int]]:
        """
        Computes grid token boundaries for a flattened sequence based on the expected pattern.
        """
        boundaries = []
        idx = 0
        for _ in range(self.num_support):
            # Input support grid block.
            boundaries.append((idx, idx + self.grid_block_size))
            idx += self.grid_block_size + 1  # Skip IO_SEP token.
            # Output support grid block.
            boundaries.append((idx, idx + self.grid_block_size))
            idx += self.grid_block_size + 1  # Skip SUP_SEP token.
        # Query grid block.
        boundaries.append((idx, idx + self.grid_block_size))
        return boundaries

    def _compute_pos_encoding(
        self, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Computes grid positional encodings for grid tokens on the fly.
        For each grid block defined in the boundaries, the 2D offset is computed as:
          row_embed + col_embed (using row-major order).
        """
        if device is None:
            device = self.row_embed.device
        grid_encodings = torch.zeros(self.max_seq_len, self.emb_size, device=device)

        # Create indices for a full grid block.
        rows = (
            torch.arange(self.num_patches_x, device=device)
            .unsqueeze(1)
            .expand(self.num_patches_x, self.num_patches_y)
            .reshape(-1)
        )
        cols = (
            torch.arange(self.num_patches_y, device=device)
            .unsqueeze(0)
            .expand(self.num_patches_x, self.num_patches_y)
            .reshape(-1)
        )
        grid_encoding = (
            self.row_embed[rows] + self.col_embed[cols]
        )  # Shape: [grid_block_size, emb_size]

        for start, end in self.grid_boundaries:
            if (end - start) != self.grid_block_size:
                raise ValueError(
                    f"Grid block from index {start} to {end} has size {end - start}, "
                    f"expected {self.grid_block_size}."
                )
            grid_encodings[start:end] = grid_encoding

        return grid_encodings.unsqueeze(0)  # Shape: [1, max_seq_len, emb_size]

    def _compute_pos_encoding_target(
        self, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Computes grid positional encodings for a target sequence consisting of an SOS token
        followed by one output grid block.
        The first token (SOS) gets no 2D offset.
        """
        if device is None:
            device = self.row_embed.device
        target_offsets = torch.zeros(self.target_seq_len, self.emb_size, device=device)

        rows = (
            torch.arange(self.num_patches_x, device=device)
            .unsqueeze(1)
            .expand(self.num_patches_x, self.num_patches_y)
            .reshape(-1)
        )
        cols = (
            torch.arange(self.num_patches_y, device=device)
            .unsqueeze(0)
            .expand(self.num_patches_x, self.num_patches_y)
            .reshape(-1)
        )
        block_offset = (
            self.row_embed[rows] + self.col_embed[cols]
        )  # Shape: [grid_block_size, emb_size]

        target_offsets[1:] = block_offset
        return target_offsets.unsqueeze(0)  # Shape: [1, target_seq_len, emb_size]

    def forward(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        """
        Adds the combined positional encoding (1D + 2D) to the input embeddings.
        Only the first x.size(1) tokens are used.
        """
        seq_len = x.size(1)
        pos1d = self.positional_1d.pos_embedding[:, :seq_len, :]
        device = x.device

        if target:
            pos2d = self._compute_pos_encoding_target(device)[:, :seq_len, :]
        else:
            pos2d = self._compute_pos_encoding(device)[:, :seq_len, :]

        return self.dropout(x + pos1d + pos2d)
