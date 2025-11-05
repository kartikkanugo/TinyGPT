"""
TinyGPT Package
===============

This module provides the essential components for building and experimenting
with a lightweight GPT-style language model.

It includes:
- Environment setup and display utilities
- File I/O helpers
- Tokenizers (regex-based and TikToken-based)
"""

from .env import env_display_tinygpt_modules
from .io_utils import io_load_text_file
from .tokenizer import RegexTokenizer, TikTokenizer
from .data_loader import tiny_data_loader
from .self_attention import (
    TinySelfAttentionQKV,
    TinySelfAttentionQKVLinear,
    TinyCausalAttention,
    TinyMultiHeadAttention,
)


__all__ = [
    # Environment utilities
    "env_display_tinygpt_modules",
    # I/O utilities
    "io_load_text_file",
    # Tokenizers
    "RegexTokenizer",
    "TikTokenizer",
    # Dataloaders
    "tiny_data_loader",
    # Self Attention
    "TinySelfAttentionQKV",
    "TinySelfAttentionQKVLinear",
    "TinyCausalAttention",
    "TinyMultiHeadAttention",
]
