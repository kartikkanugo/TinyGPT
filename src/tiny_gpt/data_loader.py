import torch

from torch.utils.data import Dataset, DataLoader

from .tokenizer import TikTokenizer


def tiny_data_loader(
    txt: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for batching and shuffling TinyDataSet samples.

    Args:
        txt (str): Raw input text.
        batch_size (int): Number of samples per batch.
        max_length (int): Maximum sequence length (context size).
        stride (int): Step size to move between windows in the text.
        shuffle (bool): Whether to shuffle samples each epoch.
        drop_last (bool, optional): Drop the last incomplete batch. Defaults to True.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.

    Returns:
        DataLoader: A PyTorch DataLoader yielding (input, target) batches for training.
    """
    # Initialize tokenizer
    tik_obj = TikTokenizer("gpt2", {"<|endoftext|>"})

    # Create dataset of input-target pairs
    data_set = _TinyDataSet(txt, tik_obj, max_length, stride)

    # Wrap dataset in a DataLoader for batching and shuffling
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return data_loader


class _TinyDataSet(Dataset):
    """
    A simple PyTorch dataset for training a small language model.

    This class takes raw text, tokenizes it, and creates overlapping
    input-target pairs using a sliding window.

    - Each `input` is a sequence of `max_length` tokens.
    - Each `target` is the same sequence shifted by one token (for next-token prediction).
    - The `stride` controls how much the window moves forward for the next sample.

    Example:
        text = "The cat sat on the mat"
        tokenizer = TikTokenizer("gpt2", {"<|endoftext|>"})
        dataset = TinyDataSet(text, tokenizer, max_length=4, stride=2)

        # if tokens = [10, 20, 30, 40, 50, 60]
        # dataset samples will be:
        # input_ids[0] = [10, 20, 30, 40]
        # target_ids[0] = [20, 30, 40, 50]
        # input_ids[1] = [30, 40, 50, 60]
        # target_ids[1] = [40, 50, 60, 70]  (if available)
    """

    def __init__(self, txt: str, tokenizer: TikTokenizer, max_length: int, stride: int):
        # Lists to store tokenized input and target sequences as tensors
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        # Tokenize the entire text into a list of token IDs
        token_ids = tokenizer.encode(txt)

        # Slide over the token sequence with the given stride
        # Each sample is a window of length `max_length`
        for i in range(0, len(token_ids) - max_length, stride):
            # Input: tokens from i to i + max_length
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))

            # Target: tokens shifted by 1 (predict the next token)
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + 1 + max_length]))

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieve one (input, target) pair by index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                input_ids  - tensor of token IDs for input sequence
                target_ids - tensor of token IDs shifted by one (the next-token labels)
        """
        return (self.input_ids[idx], self.target_ids[idx])
