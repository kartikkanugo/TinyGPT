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
