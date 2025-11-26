"""
Text generation module
"""

import torch


def tiny_generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text tokens using a simple greedy decoding strategy.

    Parameters
    ----------
    model : torch.nn.Module
        The language model that takes token indices and returns logits.
    idx : torch.Tensor
        The current sequence of token indices. Shape: (batch, seq_len).
    max_new_tokens : int
        Number of new tokens to generate.
    context_size : int
        Maximum number of previous tokens (context window) to feed into the model.

    Returns
    -------
    torch.Tensor
        The full sequence including the newly generated tokens.
    """

    for _ in range(max_new_tokens):

        # Keep only the last `context_size` tokens to respect model's context window.
        idx_cond = idx[:, -context_size:]

        # Disable gradient calculation for faster inference.
        with torch.no_grad():
            # Forward pass: obtain logits for each position in the sequence.
            logits = model(idx_cond)

        # Extract logits of the last position (the model's prediction for the next token).
        logits = logits[:, -1, :]

        # Convert logits to probabilities using softmax.
        probas = torch.softmax(logits, dim=-1)

        # Greedy decoding: pick the token with the highest probability.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append the predicted token to the sequence.
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
