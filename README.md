CS336 LLM Basics

This repository provides a compact decoder-only LLM implementation for small-scale experiments. It includes byte-level BPE, a Transformer language model, training and evaluation loops, loss functions, an optimizer, learning-rate scheduling, gradient clipping, and checkpointing. Dataset loading is intentionally left to the caller.

## Features

- Byte-level BPE training, serialization, Unicode support, and special-token encoding
- Causal multi-head attention, RoPE, KV caching, RMSNorm, and a SwiGLU feed-forward network
- Decoder-only Transformer forward passes and autoregressive generation
- Cross-entropy, causal language-modeling loss, perplexity, AdamW, and warmup-cosine scheduling
- Training and evaluation entry points for existing tensors, iterables, or DataLoader batches
- Configuration and extension points for MoE; routing and load balancing remain intentionally unimplemented and raise an explicit error if invoked

Linear projections use `torch.nn.Linear` directly instead of reimplementing basic PyTorch operators.

## Quick Start

```python
import torch

from LLM_basics import TransformerConfig, TransformerLM
from LLM_basics.optimizer import AdamW
from main import train_step

config = TransformerConfig(
    vocab_size=512,
    context_length=64,
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=128,
)
model = TransformerLM(config)
optimizer = AdamW(model.parameters(), lr=3e-4)

# Dataset loading is outside this project; pass an existing token tensor.
token_ids = torch.randint(0, config.vocab_size, (2, 33))
loss = train_step(model, token_ids, optimizer)
```

Train and load a BPE tokenizer:

```python
from LLM_basics.tokenizer import load_tokenizer_from_dir, train_bpe

train_bpe(
    "corpus.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
    save_path="tokenizer_artifacts",
)
tokenizer = load_tokenizer_from_dir("tokenizer_artifacts")
ids = tokenizer.encode("Hello, LLM!")
assert tokenizer.decode(ids) == "Hello, LLM!"
```

## Testing

The test suite uses Python's standard-library `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

