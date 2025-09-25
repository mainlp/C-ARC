from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseTokenizer(ABC):
    def __init__(self) -> None:
        self.token_to_id: Dict[Any, int] = {}
        self.id_to_token: Dict[int, Any] = {}
        self.special_tokens_to_ids: Dict[Any, Any] = {}
        self.PAD_idx: int = -1
        self.vocab_size: int = 0

    @abstractmethod
    def tokenize(
        self, grids: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Converts input data into a sequence of token IDs.

        Args:
            grid: 2D grid of integers.

        Returns:
            List[int]: Sequence of token IDs.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> List[List[Optional[int]]]:
        """
        Converts a sequence of token IDs back into the original 2D grid.

        Args:
            tokens: Sequence of token IDs.
            grid_size: The size of the output grid.

        Returns:
            Reconstructed 2D grid.
        """
        pass
