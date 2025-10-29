import re
from typing import List, Dict, Optional, Set, Union

import tiktoken


class RegexTokenizer:
    """
    A lightweight regular expression–based tokenizer that handles punctuation and spacing.

    This tokenizer:
    - Separates text into words, punctuation marks, and symbols.
    - Removes whitespace tokens from the output.
    - Creates a simple word-to-ID and ID-to-word vocabulary.
    - Supports encoding (text → token IDs) and decoding (token IDs → text).
    - Replaces missing or unknown words with the <|unk|> token.

    Example:
        >>> tokenizer = RegexTokenizer("Hello, world!")
        >>> tokens = tokenizer.split()
        >>> print(tokens)
        ['Hello', ',', 'world', '!']
        >>> vocab = tokenizer.create_vocabulary()
        >>> encoded = tokenizer.encode("Hello there!")
        >>> print(encoded)
        [0, 5, 2]  # assuming <|unk|> = 5
        >>> decoded = tokenizer.decode(encoded)
        >>> print(decoded)
        "Hello <|unk|>!"
    """

    def __init__(self, vocab_stream: str):
        """
        Initializes the tokenizer with input text.

        Args:
            file_stream (str): The raw text to tokenize.
        """
        self.f = vocab_stream
        self.vocab_splits = None
        self.vocabulary_si = None  # string → integer mapping
        self.vocabulary_is = None  # integer → string mapping

    def split(self) -> List[str]:
        """
        Splits the vocab string into tokens (words and punctuations).

        Uses regex to separate punctuation and whitespace, ensuring
        tokens like "--" are correctly handled.

        Returns:
            List[str]: A list of cleaned tokens (no whitespace tokens).
        """
        self.vocab_splits = self._split_internal(self.f)
        return self.vocab_splits

    def _split_internal(self, text: str) -> List[str]:
        """
        Internal split helper.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of stripped tokens.
        """
        # Split on punctuation, double-dash, or whitespace
        result = re.split(r"([,.:;?_!\"()']|--|\s)", text)

        # Remove empty and purely whitespace tokens
        result = [item.strip() for item in result if item.strip()]

        return result

    def create_vocabulary(self) -> Dict[str, int]:
        """
        Creates a vocabulary mapping for the tokens.

        Returns:
            Dict[str, int]: A dictionary mapping token → integer ID.
        """
        # Sort and remove duplicates
        word_set = sorted(set(self.vocab_splits)) + ["<|endoftext|>", "<|unk|>"]

        # Create both token→id and id→token mappings
        self.vocabulary_si = {token: idx for idx, token in enumerate(word_set)}
        self.vocabulary_is = {idx: token for idx, token in enumerate(word_set)}

        return self.vocabulary_si

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text into a sequence of token IDs.

        Unknown tokens are replaced by <|unk|>.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of token IDs.
        """
        word_tokens = self._split_internal(text)

        # Replace tokens not found in vocabulary with <|unk|>
        word_tokens_unk = [
            w if w in self.vocabulary_si else "<|unk|>" for w in word_tokens
        ]

        # Map each token to its corresponding ID
        ids = [self.vocabulary_si[w] for w in word_tokens_unk]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into readable text.

        Args:
            ids (List[int]): The list of token IDs.

        Returns:
            str: The reconstructed text.
        """
        # Join tokens with spaces
        text = " ".join([self.vocabulary_is[i] for i in ids])

        # Remove unnecessary spaces before punctuation
        # Example: "Hello , world !" → "Hello, world!"
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)

        return text


import tiktoken
from typing import List, Optional, Union, Set


class TikTokenizer:
    """
    Wrapper around TikToken to provide a clean interface
    and optional preprocessing logic.

    Parameters
    ----------
    model_name : str, optional
        Name of the TikToken model encoding to use (default is "gpt2").
    allowed_special : set[str] | "all"|None, optional
        Set or mode of allowed special tokens.
        - If None: no special tokens allowed.
        - If "all": all TikToken special tokens are allowed.
        - If set[str]: only those tokens are allowed.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        allowed_special: Optional[Set[str] | str] = None,
    ):
        self.enc = tiktoken.get_encoding(model_name)
        self.allowed_special = allowed_special

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        if self.allowed_special is None:
            return self.enc.encode(text)
        elif isinstance(self.allowed_special, set | str):
            return self.enc.encode(text, allowed_special=self.allowed_special)
        else:
            raise TypeError("allowed_special must be None or a set[str]")

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        return self.enc.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text."""
        if self.allowed_special is None:
            return len(self.enc.encode(text))
        elif isinstance(self.allowed_special, set | str):
            return len(self.enc.encode(text, allowed_special=self.allowed_special))
        else:
            raise TypeError("allowed_special must be None or a set[str]")

    def get_n_vocab_tokens(self) -> int:
        """Returns the number of tokens in vocab example gpt2"""
        return self.enc.n_vocab
