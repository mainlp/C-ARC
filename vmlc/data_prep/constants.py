# Constants for special tokens
SOS_TOKEN = "SOS"  # Start of Sentence token
EOS_TOKEN = "EOS"  # End of Sentence token
PAD_TOKEN = "PAD"  # Padding token
IO_SEP = "IO_SEP"  # Separator for Input -> Output in support examples
SUP_SEP = "SUP_SEP"  # Separator for multiple support examples in input sequence

# Default input and output symbols
INPUT_SYMBOLS_DEFAULT = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
OUTPUT_SYMBOLS_DEFAULT = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, IO_SEP, SUP_SEP]
SPECIAL_TOKENS_TO_IDX = {
    token: i + len(OUTPUT_SYMBOLS_DEFAULT) for i, token in enumerate(SPECIAL_TOKENS)
}
