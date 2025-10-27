# 🧠 TinyGPT

A from-scratch, minimal GPT-like model built for learning and experimentation.  
Inspired by Sebastian Raschka’s _Build a Large Language Model (From Scratch)_ and Stanford’s deep learning courses.

---


## TinyGPT — Python Module

TinyGPT is a Python module that provides a collection of helper functions and classes designed to build and experiment with lightweight large language models (LLMs).  
It serves as the foundational layer for model components such as tokenization, training, and inference.


## 🎯 Project Goals

- Learn the **core principles** behind transformer-based language models.
- Build each component of GPT step-by-step in **PyTorch**.
- Train a small model that runs comfortably on a **single RTX 4070 GPU**.
- Develop a clean and reproducible **training + inference pipeline**.

---

## 🚀 Milestones & Targets

### 🧩 **Stage 1 – Setup & Environment**

- [x] Create project structure and setup Python virtual environment.
- [x] Load a small sample dataset (e.g., TinyStories, Shakespeare, or custom text).

### 🧠 **Stage 2 – Tokenization & Data Pipeline**

- [ ] Implement a basic **character-level tokenizer** (encode/decode).
- [ ] Extend to a **byte-pair encoding (BPE)** tokenizer later.
- [ ] Split data into **train/validation** sets.
- [ ] Create a **PyTorch Dataset & DataLoader** for batching.

### ⚙️ **Stage 3 – Transformer Core Components**

- [ ] Implement **positional embeddings**.
- [ ] Implement **multi-head self-attention** mechanism from scratch.
- [ ] Add **feed-forward (MLP)** block and **LayerNorm**.
- [ ] Stack multiple transformer blocks to form the GPT backbone.

### 🔥 **Stage 4 – Training Loop**

- [ ] Write training loop with **cross-entropy loss** and **AdamW optimizer**.
- [ ] Add **gradient clipping** and **learning rate scheduler**.
- [ ] Implement checkpoint saving & resuming.
- [ ] Log training/validation loss and visualize progress (matplotlib or wandb).

### 💬 **Stage 5 – Text Generation**

- [ ] Implement sampling (`temperature`, `top-k`, `top-p` decoding).
- [ ] Generate text from seed prompts and tune parameters.
- [ ] Create a simple CLI interface for text generation.

### ⚡ **Stage 6 – Optimization & Experiments**

- [ ] Profile GPU memory and optimize batch sizes.
- [ ] Experiment with smaller/larger model configs.
- [ ] Try mixed-precision training (FP16) on RTX 4070.
- [ ] Compare training speed between CPU and GPU.

### 🧪 **Stage 7 – Extensions (Optional but Fun)**

- [ ] Implement a **BPE tokenizer** (from Hugging Face or custom).
- [ ] Add **configurable model hyperparameters** via JSON or argparse.
- [ ] Integrate a small **web demo (Gradio)** for chat-like interface.
- [ ] Explore fine-tuning on a domain-specific dataset (e.g., physics text).

---

## 🧰 Tech Stack

- **Language:** Python 3.13
- **Framework:** PyTorch
- **GPU:** NVIDIA RTX 4070
- **Libraries:** NumPy, tqdm, matplotlib, sentencepiece (later), Gradio (optional)

---

## 📘 References

- [Sebastian Raschka – _Build a Large Language Model (From Scratch)_](https://sebastianraschka.com/llms-from-scratch/)
- [Stanford CS324: Large Language Models](https://online.stanford.edu/courses/cs336-language-modeling-scratch)
- [Andrej Karpathy – nanoGPT](https://github.com/karpathy/nanoGPT)

---

## 🧭 Next Steps

- Focus on understanding **attention mechanics** before scaling up.
- Keep all training runs small (e.g., context size ≤ 256, vocab ≤ 2000).
- Commit early, commit often — show incremental progress in code and README.
