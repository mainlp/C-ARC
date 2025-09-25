"""
This file contains code derived and modified from the following source:

Original source:
Lake, B. M. and Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623, 115-121.
<https://github.com/brendenlake/MLC/blob/main/model.py>

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

from typing import Tuple

import torch
import torch.nn as nn

from vmlc.model.positional_encodings import LearnedPositionalEncodingCombined


class SeqToSeqTransformer(nn.Module):
    """
    Transformer implementation for sequence-to-sequence task.

    Args:
        hidden_size (int): Size of the hidden layer in the transformer.
        output_size (int): Size of the output layer.
        PAD_idx_input (int): Padding index for the input sequences.
        PAD_idx_output (int): Padding index for the output sequences.
        nlayers_encoder (int, optional): Number of layers in the transformer encoder. Defaults to 5.
        nlayers_decoder (int, optional): Number of layers in the transformer decoder. Defaults to 3.
        nhead (int, optional): Number of attention heads in the transformer. Defaults to 8.
        dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        ff_mult (int, optional): Multiplier for the feedforward layer size. Defaults to 4.
        activation (str, optional): Activation function ('gelu' or 'relu'). Defaults to 'gelu'.
        grid_size (Tuple[int, int]): Size of the grid (e.g. (10, 10)).
        patch_size (Tuple[int, int]): Size of each patch (e.g. (2, 2)).
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        PAD_idx_input: int,
        PAD_idx_output: int,
        nlayers_encoder: int = 5,
        nlayers_decoder: int = 3,
        nhead: int = 8,
        dropout_p: float = 0.1,
        ff_mult: int = 4,
        activation: str = "gelu",
        grid_size: Tuple[int, int] = (10, 10),
        patch_size: Tuple[int, int] = (2, 2),
        num_support: int = 18,
    ):
        super(SeqToSeqTransformer, self).__init__()

        assert activation in ["gelu", "relu"]
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.PAD_idx_input = PAD_idx_input
        self.PAD_idx_output = PAD_idx_output
        self.nlayers_encoder = nlayers_encoder
        self.nlayers_decoder = nlayers_decoder
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.dim_feedforward = hidden_size * ff_mult
        self.act = activation

        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=nlayers_encoder,
            num_decoder_layers=nlayers_decoder,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout_p,
            batch_first=True,
            activation=activation,
        )

        num_patches_x = grid_size[0] // patch_size[0]
        num_patches_y = grid_size[1] // patch_size[1]

        self.positional_encoding = LearnedPositionalEncodingCombined(
            emb_size=hidden_size,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            num_support=num_support,
            dropout=dropout_p,
        )

        self.input_embedding = nn.Embedding(output_size, hidden_size)
        self.output_embedding = nn.Embedding(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def prep_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepares the input sequences by applying positional embeddings.

        Args:
            x (torch.Tensor): Input tensor of token IDs. Shape: [batch_size, seq_len].

        Returns:
            torch.Tensor: Encoded tensor with positional embeddings of shape [batch_size, seq_len, hidden_size].
        """
        assert (
            not (x == self.PAD_idx_input).any().item()
        ), f"Found PAD token {self.PAD_idx_input} in input!"
        x_embed = self.input_embedding(x)  # [batch_size, seq_len, hidden_size]
        src_embed = self.positional_encoding(x_embed)
        return src_embed

    def prep_decode(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares the target sequences by embedding them and creating masks for the transformer decoder.

        Args:
            tgt (torch.Tensor): Target tensor of token IDs with shape [batch_size, maxlen_tgt].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the target embeddings and target mask.
        """
        assert (
            not (tgt == self.PAD_idx_output).any().item()
        ), f"Found PAD token {self.PAD_idx_output} in output!"
        maxlen_tgt = tgt.size(1)
        z_embed = self.output_embedding(tgt)
        tgt_embed = self.positional_encoding(z_embed, target=True)
        tgt_mask = (
            self.transformer.generate_square_subsequent_mask(maxlen_tgt)
            .to(tgt.device)
            .bool()
        )

        return tgt_embed, tgt_mask

    def forward(self, tgt: torch.Tensor, batch: dict) -> torch.Tensor:
        """
        Forward pass through the entire model (encoder and decoder).

        Args:
            z_padded (torch.Tensor): Padded target tensor of token IDs with shape [batch_size, maxlen_tgt].
            batch (dict): Dictionary containing the input tensor and other batch-related information.

        Returns:
            torch.Tensor: The output tensor after passing through the transformer and final linear layer.
        """
        x = batch["xq_context"]
        src_embed = self.prep_encode(x)
        tgt_embed, tgt_mask = self.prep_decode(tgt)
        trans_out = self.transformer(
            src_embed,
            tgt_embed,
            tgt_mask=tgt_mask,
        )
        output = self.out(trans_out)
        return output

    def encode(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through the encoder only.

        Args:
            batch (dict): Dictionary containing the input tensor and other batch-related information.

        Returns:
            torch.Tensor: Encoded memory tensor.
        """
        xq_context = batch["xq_context"]
        src_embed = self.prep_encode(xq_context)
        memory = self.transformer.encoder(src_embed)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder only.

        Args:
            tgt (torch.Tensor): Target tensor of token IDs with shape [batch_size, maxlen_tgt].
            memory (torch.Tensor): Output tensor from the transformer encoder.

        Returns:
            torch.Tensor: Output tensor from the transformer decoder.
        """
        tgt_embed, tgt_mask = self.prep_decode(tgt)
        trans_out = self.transformer.decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
        )
        output = self.out(trans_out)
        return output
