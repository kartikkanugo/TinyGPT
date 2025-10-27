import platform
import sys

import numpy as np
import pandas as pd
import tiktoken
import torch

# --- Public API ---


def env_display_tinygpt_modules():
    """Run all setup checks together."""

    print(f"Running {env_display_tinygpt_modules.__name__}")
    _env_summary()
    _cuda_info()
    _test_tokenizer()
    print("✅ TinyGPT environment initialized successfully!")
    print("-" * 50)


# --- Private Helpers ---


def _env_summary():
    """Print Python, Torch, NumPy, and tiktoken version details."""
    print("🔧 Environment Summary")
    print("-" * 50)
    print(f"Python version : {platform.python_version()}")
    print(f"Platform       : {platform.system()} {platform.release()}")
    print()
    print(f"Torch version  : {torch.__version__}")
    print(f"NumPy version  : {np.__version__}")
    print(f"Tiktoken ver.  : {tiktoken.__version__}")
    if "pandas" in sys.modules:
        print(f"Pandas version : {pd.__version__}")
    print("-" * 50)


def _cuda_info():
    """Check if CUDA is available and show GPU details."""
    print("⚙️  CUDA & GPU Information")
    print("-" * 50)
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"🧠 GPU name      : {torch.cuda.get_device_name(0)}")
        print(
            f"💽 Total memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("❌ CUDA not available — running on CPU")
    print("-" * 50)


def _test_tokenizer(sample_text="Once upon a time in TinyGPT..."):
    """Quick test for tiktoken tokenizer functionality."""
    print("🔤 Tokenizer Test (tiktoken)")
    print("-" * 50)
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(sample_text)
    print(f"Input text : {sample_text}")
    print(f"Tokens     : {tokens}")
    print(f"Decoded    : {enc.decode(tokens)}")
    print("-" * 50)
