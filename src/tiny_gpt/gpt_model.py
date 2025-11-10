"""
This module creates gpt modules

"""

import torch
import torch.nn as nn


class TinyLayerNorm(nn.Module):
    """
    A simplified implementation of Layer Normalization.

    Layer Normalization normalizes the input across the last dimension
    (i.e., across features of each token in a sequence), ensuring
    that the activations have zero mean and unit variance. This helps
    stabilize training and improve convergence in neural networks.

    Attributes
    ----------
    eps : float
        Small constant added to the variance for numerical stability.
    scale : torch.nn.Parameter
        Learnable parameter that scales the normalized output (gamma).
    shift : torch.nn.Parameter
        Learnable parameter that shifts the normalized output (beta).
    """

    def __init__(self, emb_dim: int):
        """
        Initialize the TinyLayerNorm module.

        Parameters
        ----------
        emb_dim : int
            The dimensionality of the input embeddings (number of features per token).
        """
        super().__init__()
        self.eps = 1e-5  # Prevents division by zero in variance normalization
        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )  # Learnable scaling factor (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Learnable bias/shift (beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., emb_dim), where normalization is applied
            over the last dimension.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as input.
        """
        # Compute mean and variance along the last dimension (feature axis)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize input to have zero mean and unit variance
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift the normalized output using learnable parameters
        return self.scale * norm_x + self.shift
