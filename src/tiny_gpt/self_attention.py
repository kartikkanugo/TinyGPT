"""
Self Attention Module
"""

import torch
import torch.nn as nn


class TinySelfAttentionQKV(nn.Module):
    """
    A tiny implementation of self-attention using separate Query, Key, Value projections.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        # -----------------------------
        # Learnable weight matrices for Query, Key, and Value
        # d_in: input dimension of token embeddings
        # d_out: output dimension of the attention space
        # -----------------------------
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))  # maps input to queries
        self.w_key = nn.Parameter(torch.rand(d_in, d_out))  # maps input to keys
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))  # maps input to values

    def forward(self, x):
        """
        x: input tensor of shape (seq_len, d_in), where seq_len = number of tokens
        Returns:
            context_vec: self-attention output of shape (seq_len, d_out)
        """
        # -----------------------------
        # Linear projections to get queries, keys, values
        # Each token embedding is projected into Q, K, V spaces
        # -----------------------------
        queries = x @ self.w_query  # shape: (seq_len, d_out)
        keys = x @ self.w_key  # shape: (seq_len, d_out)
        values = x @ self.w_value  # shape: (seq_len, d_out)

        # -----------------------------
        # Compute attention scores (compatibility between queries and keys)
        # Dot product of queries and keys^T
        # -----------------------------
        attn_scores = queries @ keys.T  # shape: (seq_len, seq_len)

        # -----------------------------
        # Scale attention scores to avoid large values causing softmax saturation
        # Then apply softmax to get attention weights
        # Each row sums to 1
        # -----------------------------
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # -----------------------------
        # Compute context vectors
        # Weighted sum of the value vectors based on attention weights
        # Each token's representation is updated based on other tokens
        # -----------------------------
        context_vec = attn_weights @ values  # shape: (seq_len, d_out)

        return context_vec


class TinySelfAttentionQKVLinear(nn.Module):
    """
    A tiny implementation of self-attention using separate Query, Key, and Value
    projections.

    We can improve the SelfAttention_v1 implementation further by utilizing PyTorch's
    nn.Linear layers, which efficiently perform matrix multiplication when the bias
    units are disabled. Additionally, a major advantage of using nn.Linear instead of
    manually implementing nn.Parameter(torch.rand(...)) is that nn.Linear provides
    optimized weight initialization, contributing to more stable and effective model
    training.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # -----------------------------
        # Learnable weight matrices for Query, Key, and Value
        # d_in: input dimension of token embeddings
        # d_out: output dimension of the attention space
        # -----------------------------
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # maps input to queries
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # maps input to keys
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # maps input to values

    def forward(self, x):
        """
        x: input tensor of shape (seq_len, d_in), where seq_len = number of tokens
        Returns:
            context_vec: self-attention output of shape (seq_len, d_out)
        """
        # -----------------------------
        # Linear projections to get queries, keys, values
        # Each token embedding is projected into Q, K, V spaces
        # -----------------------------
        queries = self.w_query(x)  # shape: (seq_len, d_out)
        keys = self.w_key(x)  # shape: (seq_len, d_out)
        values = self.w_value(x)  # shape: (seq_len, d_out)

        # -----------------------------
        # Compute attention scores (compatibility between queries and keys)
        # Dot product of queries and keys^T
        # -----------------------------
        attn_scores = queries @ keys.T  # shape: (seq_len, seq_len)

        # -----------------------------
        # Scale attention scores to avoid large values causing softmax saturation
        # Then apply softmax to get attention weights
        # Each row sums to 1
        # -----------------------------
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # -----------------------------
        # Compute context vectors
        # Weighted sum of the value vectors based on attention weights
        # Each token's representation is updated based on other tokens
        # -----------------------------
        context_vec = attn_weights @ values  # shape: (seq_len, d_out)

        return context_vec


class TinyCausalAttention(nn.Module):
    """
    Causal Attention
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        # Linear projections for Query, Key, and Value
        # Each maps from input dimension (d_in) to output dimension (d_out)
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout layer for regularization in attention weights
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular matrix of ones)
        # Ensures that each token can only attend to previous tokens (and itself),
        # preventing information leakage from the future.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        x: input tensor of shape (seq_len, d_in), where seq_len = number of tokens
        Returns:
            context_vec: self-attention output of shape (seq_len, d_out)
        """
        # x: (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        # Compute key, query, and value projections
        # Each will have shape: (batch_size, num_tokens, d_out)
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        # Compute raw attention scores by dot product between queries and keys
        # Result shape: (batch_size, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(1, 2)

        # Apply causal mask: set future positions to -inf
        # This ensures softmax assigns zero probability to future tokens
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Normalize scores into probabilities using softmax
        # Divide by sqrt(d_out) for stable gradients (scaling trick)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)

        # Compute context vector as weighted sum of value vectors
        # Shape: (batch_size, num_tokens, d_out)
        context_vec = attn_weights @ values

        # Return the final attention output
        return context_vec


class TinyMultiHeadAttention(nn.Module):
    """
    A lightweight implementation of Multi-Head Self-Attention with causal masking.

    This module splits the input embeddings into multiple attention heads, performs
    scaled dot-product attention independently for each head, and then concatenates
    the results back together. A causal mask ensures that each position can only
    attend to current and previous tokens (no future information leakage).

    Args:
        d_in (int): Input feature dimension.
        d_out (int): Output feature dimension. Must be divisible by num_heads.
        context_length (int): Maximum sequence length for attention masking.
        dropout (float): Dropout probability applied to attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add bias to Q, K, V projections. Default is False.

    Shape:
        - Input:  (batch_size, seq_len, d_in)
        - Output: (batch_size, seq_len, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Ensure that output dimension can be evenly split across all heads
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimensionality of each attention head

        # Linear projections to generate Queries, Keys, and Values
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final linear projection after concatenating all heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout applied to attention weights
        self.dropout = nn.Dropout(dropout)

        # Register a causal mask (upper-triangular)
        # Prevents tokens from attending to future positions
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass of multi-head causal attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_out).
        """
        b, num_tokens, d_in = x.shape

        # 1. Project input embeddings into Query, Key, and Value tensors
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        # 2. Reshape to separate attention heads
        # Shape: (batch_size, seq_len, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3. Transpose to put heads before sequence dimension
        # New shape: (batch_size, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. Compute scaled dot-product attention scores
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(2, 3)

        # 5. Apply causal mask to block attention to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 6. Normalize scores into attention weights using softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # 7. Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # 8. Compute weighted sum of values (context vectors)
        # Shape after multiplication: (batch_size, num_heads, seq_len, head_dim)
        context_vec = attn_weights @ values

        # 9. Bring head dimension back after concatenation
        # Shape: (batch_size, seq_len, num_heads * head_dim) == (b, seq_len, d_out)
        context_vec = (
            context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        )

        # 10. Apply final output projection
        context_vec = self.out_proj(context_vec)

        return context_vec
