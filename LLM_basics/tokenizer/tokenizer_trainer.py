"""Backward-compatible exports for BPE training helpers.

Historically this module was empty while the implementation lived in
``tokenizer.py``.  Keeping these re-exports allows either import style without
duplicating the stateful training algorithm.
"""

from .tokenizer import (
    PAT,
    init_vocab,
    pre_tokenize,
    pre_tokenize_string_worker,
    split_by_special_tokens,
    train_bpe,
    update_vocab,
)

__all__ = [
    "PAT",
    "init_vocab",
    "pre_tokenize",
    "pre_tokenize_string_worker",
    "split_by_special_tokens",
    "train_bpe",
    "update_vocab",
]
