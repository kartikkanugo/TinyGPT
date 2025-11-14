"""
This module creates gpt modules

"""

from .self_attention import TinyMultiHeadAttention

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


class TinyGELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    This is an activation commonly used in Transformer models.
    It applies the approximate formulation:
        GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Apply GELU activation on the input tensor.
        """
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class TinyFeedForward(nn.Module):
    """
    Transformer Feed-Forward Network (FFN).

    Structure:
        Linear -> GELU -> Linear

    Expands embedding dimension by 4x and then projects back.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (dict): Contains 'emb_dim' and other model configuration.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand dimension
            TinyGELU(),  # Non-linear activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Project back down
        )

    def forward(self, x):
        """
        Forward pass through FFN.
        """
        return self.layers(x)


class TinyTransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
    - LayerNorm
    - Multi-Head Attention
    - Residual connection
    - LayerNorm
    - FeedForward network
    - Residual connection
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (dict): Transformer configuration dictionary.
        """
        super().__init__()

        self.att = TinyMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = TinyFeedForward(cfg)
        self.norm1 = TinyLayerNorm(cfg["emb_dim"])
        self.norm2 = TinyLayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass for a transformer block.

        Residual Path 1:
            x = x + Dropout(Attention(LayerNorm(x)))

        Residual Path 2:
            x = x + Dropout(FeedForward(LayerNorm(x)))
        """
        # --- Residual Block 1 (Self-Attention) ---
        shortcut = x  # Save input for residual
        x = self.norm1(x)  # Pre-norm
        x = self.att(x)  # Multi-Head Attention
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # Residual add

        # --- Residual Block 2 (FeedForward) ---
        shortcut = x  # Save input for residual
        x = self.norm2(x)  # Pre-norm
        x = self.ff(x)  # Feed-forward network
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # Residual add

        return x


class TinyGPTModel(nn.Module):
    """
    A minimal GPT-style Transformer language model.

    Components:
        - Token embeddings
        - Positional embeddings
        - Dropout
        - N transformer blocks
        - Final LayerNorm
        - Linear output projection (LM head)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (dict): Model configuration containing:
                        vocab_size, emb_dim, context_length,
                        n_layers, n_heads, drop_rate, qkv_bias.
        """
        super().__init__()

        # Embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TinyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final normalization + output projection
        self.final_norm = TinyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass of the TinyGPT model.

        Args:
            in_idx (Tensor): LongTensor of token indices, shape (B, T)

        Returns:
            logits (Tensor): Raw output scores, shape (B, T, vocab_size)
        """
        batch_size, seq_len = in_idx.shape

        # Token lookup
        tok_embeds = self.tok_emb(in_idx)

        # Positional embeddings for sequence length
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Add token + positional embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Pass through N transformer blocks
        x = self.trf_blocks(x)

        # Final layer norm + output projection
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
