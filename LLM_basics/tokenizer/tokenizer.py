"""Byte-level BPE training and tokenization.

The implementation deliberately keeps the public API small: :func:`train_bpe`
builds a vocabulary and ordered merge list, while :class:`BPETokenizer` applies
those merges for encoding and reverses the byte representation for decoding.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import regex as re
from tqdm import tqdm

from .merge_fn import (
    build_pair_heap,
    build_pair_to_words,
    merge_pairs_with_heap_index,
    pop_most_frequent_pair,
)
from .utils import (
    find_chunk_boundaries,
    save_vocab_and_merges,
    text_to_token_bytes,
)


# GPT-2's pre-tokenization expression. BPE merges never cross the boundaries
# produced by this pattern (nor special-token boundaries).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = min(4, os.cpu_count() or 1)

Word = tuple[int, ...]


def _validate_special_tokens(special_tokens: Sequence[str] | None) -> list[str]:
    """Return a defensive copy after validating special-token invariants."""

    if special_tokens is None:
        return []

    result = list(special_tokens)
    if any(not isinstance(token, str) for token in result):
        raise TypeError("special_tokens must contain only strings")
    if any(token == "" for token in result):
        raise ValueError("special tokens must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("special_tokens must not contain duplicates")
    return result


def init_vocab(special_tokens: Sequence[str] | None = None) -> dict[int, bytes]:
    """Create the 256 byte tokens followed by the requested special tokens."""

    normalized_specials = _validate_special_tokens(special_tokens)
    vocab = {token_id: bytes((token_id,)) for token_id in range(256)}

    seen = set(vocab.values())
    for token in normalized_specials:
        token_bytes = token.encode("utf-8")
        if token_bytes in seen:
            raise ValueError(
                f"special token {token!r} duplicates an existing byte token"
            )
        vocab[len(vocab)] = token_bytes
        seen.add(token_bytes)

    return vocab


def update_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    """Append the concatenation represented by ``pair`` and return its ID."""

    left_id, right_id = pair
    if left_id not in vocab or right_id not in vocab:
        raise KeyError(f"merge pair {pair!r} refers to an unknown token ID")

    # train_bpe constructs contiguous IDs, so this path is O(1). Fall back to
    # max()+1 only for a caller-provided mapping whose IDs contain gaps.
    new_id = len(vocab)
    if new_id in vocab:
        new_id = max(vocab, default=-1) + 1
    vocab[new_id] = vocab[left_id] + vocab[right_id]
    return new_id


def split_by_special_tokens(
    text: str,
    special_tokens: Sequence[str] | None,
    include_special: bool = False,
) -> list[str]:
    """Split ``text`` on exact special-token occurrences.

    Tokens are sorted longest-first so overlapping specials such as ``<x>``
    and ``<x><y>`` have deterministic, intuitive behavior.
    """

    normalized_specials = _validate_special_tokens(special_tokens)
    if not normalized_specials:
        return [text]

    ordered = sorted(normalized_specials, key=lambda token: (-len(token), token))
    pattern = "|".join(re.escape(token) for token in ordered)
    if include_special:
        pattern = f"({pattern})"
    return re.split(pattern, text)


def pre_tokenize(
    string: str,
    special_tokens: Sequence[str] | None = None,
    including_special: bool = False,
) -> Counter[Word]:
    """Count UTF-8 byte sequences produced by regex pre-tokenization.

    During BPE training ``including_special`` should remain false: special
    tokens already have fixed vocabulary IDs and must not influence merges.
    The optional true mode is retained for callers that want to inspect every
    segment, and represents a special token by its raw UTF-8 bytes.
    """

    normalized_specials = _validate_special_tokens(special_tokens)
    special_set = set(normalized_specials)
    word_counter: Counter[Word] = Counter()

    for chunk in split_by_special_tokens(
        string,
        normalized_specials,
        include_special=including_special,
    ):
        if not chunk:
            continue
        if including_special and chunk in special_set:
            word_counter[tuple(chunk.encode("utf-8"))] += 1
            continue

        for match in re.finditer(PAT, chunk):
            word_counter[tuple(match.group(0).encode("utf-8"))] += 1

    return word_counter


def _pre_tokenize_file_chunk(
    input_path: str,
    special_tokens: Sequence[str],
    start: int,
    end: int,
) -> Counter[Word]:
    """Read and pre-tokenize one byte-aligned corpus range."""

    with open(input_path, "rb") as corpus_file:
        corpus_file.seek(start)
        data = corpus_file.read(end - start)

    # Chunk boundaries are placed on a complete UTF-8 encoded delimiter, so a
    # strict decode is preferable to silently dropping corrupt corpus bytes.
    text = data.decode("utf-8")
    return pre_tokenize(text, special_tokens)


def pre_tokenize_string_worker(*args: Any) -> None:
    """Compatibility wrapper for the original queue-based worker API."""

    input_path, special_tokens, queue, start, end, include_special = args
    with open(input_path, "rb") as corpus_file:
        corpus_file.seek(start)
        text = corpus_file.read(end - start).decode("utf-8")
    queue.put(pre_tokenize(text, special_tokens, include_special))


def _safe_parallel_delimiter(special_tokens: Sequence[str]) -> bytes | None:
    """Choose a special token whose byte occurrences are safe chunk boundaries.

    A raw occurrence is unsafe when it can sit inside a longer special token or
    overlap a special token that starts earlier. For instance, splitting every
    ``aba`` occurrence in ``ababa`` would choose the second, overlapping match,
    although regex splitting consumes only the first. This conservative check
    falls back to one chunk whenever such an ambiguity is possible.
    """

    encoded_tokens = [token.encode("utf-8") for token in special_tokens]
    for candidate in sorted(encoded_tokens, key=len, reverse=True):
        if any(
            candidate != other and candidate in other
            for other in encoded_tokens
        ):
            continue

        has_incoming_overlap = False
        for other in encoded_tokens:
            max_overlap = min(len(other) - 1, len(candidate) - 1)
            if any(
                other[-overlap:] == candidate[:overlap]
                for overlap in range(1, max_overlap + 1)
            ):
                has_incoming_overlap = True
                break
        if not has_incoming_overlap:
            return candidate
    return None


def _count_corpus_pretokens(
    input_path: str | os.PathLike[str],
    special_tokens: Sequence[str],
    desired_num_chunks: int,
) -> Counter[Word]:
    """Pre-tokenize a corpus, parallelizing independent delimiter ranges."""

    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")

    path = os.fspath(input_path)
    # Only a removed, non-overlapping special token is a guaranteed regex
    # boundary. A newline is not generally equivalent because PAT may group
    # consecutive whitespace across it. Ambiguous specials likewise fall back
    # to one exact range; correctness takes precedence over parallelism.
    safe_delimiter = _safe_parallel_delimiter(special_tokens)
    if safe_delimiter is not None:
        delimiter = safe_delimiter
        boundary_chunks = desired_num_chunks
    else:
        delimiter = b"\n"  # unused as an internal boundary when chunks == 1
        boundary_chunks = 1
    with open(path, "rb") as corpus_file:
        boundaries = find_chunk_boundaries(
            corpus_file,
            desired_num_chunks=boundary_chunks,
            split_special_token=delimiter,
        )

    ranges = [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]
    if not ranges:
        return Counter()
    if len(ranges) == 1:
        start, end = ranges[0]
        return _pre_tokenize_file_chunk(path, special_tokens, start, end)

    word_counter: Counter[Word] = Counter()
    with ThreadPoolExecutor(max_workers=min(len(ranges), NUM_PROCESSES)) as pool:
        futures = [
            pool.submit(
                _pre_tokenize_file_chunk,
                path,
                special_tokens,
                start,
                end,
            )
            for start, end in ranges
        ]
        # Consume results in corpus order. Counter addition is commutative, but
        # ordered collection also makes exception reporting deterministic.
        for future in futures:
            word_counter.update(future.result())
    return word_counter


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a deterministic byte-level BPE vocabulary.

    Pair-frequency ties are resolved by the lexicographically greatest pair of
    byte strings, matching the CS336 reference convention. If the corpus runs
    out of adjacent pairs, training stops early instead of inventing tokens.

    Extra keyword arguments:
        desired_num_chunks: number of pre-tokenization ranges (default: up to 4)
        save_path: optional directory for ``vocab.json`` and ``merges.txt``
    """

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")

    normalized_specials = _validate_special_tokens(special_tokens)
    vocab = init_vocab(normalized_specials)
    if vocab_size < len(vocab):
        raise ValueError(
            "vocab_size must be at least 256 plus the number of special tokens"
        )

    desired_num_chunks = kwargs.pop("desired_num_chunks", NUM_PROCESSES)
    save_path = kwargs.pop("save_path", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected train_bpe keyword argument(s): {unknown}")
    if isinstance(desired_num_chunks, bool) or not isinstance(
        desired_num_chunks, int
    ):
        raise TypeError("desired_num_chunks must be an integer")

    word_counter = _count_corpus_pretokens(
        input_path,
        normalized_specials,
        desired_num_chunks,
    )
    if verbose:
        print(
            f"Pre-tokenization complete: {len(word_counter)} unique segments, "
            f"{sum(word_counter.values())} total segments."
        )

    pair_counter: Counter[tuple[int, int]] = Counter()
    for word, frequency in word_counter.items():
        for left, right in zip(word, word[1:]):
            pair_counter[(left, right)] += frequency
    pair_to_words = build_pair_to_words(word_counter)
    pair_heap = build_pair_heap(pair_counter, vocab)

    merges: list[tuple[bytes, bytes]] = []
    target_merges = vocab_size - len(vocab)
    for merge_index in range(target_merges):
        try:
            most_frequent_pair = pop_most_frequent_pair(
                pair_heap,
                pair_counter,
            )
        except ValueError:
            if verbose:
                print(
                    f"Stopped after {merge_index} merges: the corpus has no "
                    "remaining adjacent token pairs."
                )
            break

        left_id, right_id = most_frequent_pair
        merges.append((vocab[left_id], vocab[right_id]))
        new_id = update_vocab(vocab, most_frequent_pair)
        word_counter, pair_counter, pair_heap, pair_to_words = (
            merge_pairs_with_heap_index(
                word_counter,
                pair_counter,
                most_frequent_pair,
                new_id,
                vocab,
                pair_heap,
                pair_to_words,
            )
        )

    if save_path is not None:
        save_vocab_and_merges(vocab, merges, save_path)
        output_path = Path(save_path)

        # A line-oriented file cannot represent a token containing CR or LF
        # without changing its value.  JSON is therefore the canonical format
        # for special tokens, including the common line-free case.
        special_json_path = output_path / "special_tokens.json"
        special_json_path.write_text(
            json.dumps(normalized_specials, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Keep producing the historical file when it is lossless so older
        # callers can continue to consume newly trained tokenizers.  Remove a
        # stale legacy file if this directory is reused with newline tokens.
        legacy_special_path = output_path / "special_tokens.txt"
        legacy_is_lossless = all(
            "\r" not in token and "\n" not in token
            for token in normalized_specials
        )
        if legacy_is_lossless:
            legacy_special_path.write_text(
                "".join(f"{token}\n" for token in normalized_specials),
                encoding="utf-8",
            )
        elif legacy_special_path.exists():
            legacy_special_path.unlink()

    return vocab, merges


class BPETokenizer:
    """Encode and decode text with an ordered byte-level BPE merge table."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Sequence[tuple[bytes, bytes]],
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        normalized_specials = _validate_special_tokens(special_tokens)
        self.vocab = dict(vocab)

        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in self.vocab
        ):
            raise TypeError("vocabulary IDs must be integers")
        if any(not isinstance(token, bytes) for token in self.vocab.values()):
            raise TypeError("vocabulary values must be bytes")
        missing_byte_ids = [
            token_id
            for token_id in range(256)
            if self.vocab.get(token_id) != bytes((token_id,))
        ]
        if missing_byte_ids:
            raise ValueError(
                "vocab must map IDs 0..255 to their corresponding single bytes"
            )

        # A tokenizer loaded from a base vocabulary may receive additional
        # runtime specials. Add them without mutating the caller's dictionary.
        next_id = max(self.vocab, default=-1) + 1
        existing_bytes = set(self.vocab.values())
        for token in normalized_specials:
            encoded = token.encode("utf-8")
            if encoded not in existing_bytes:
                self.vocab[next_id] = encoded
                existing_bytes.add(encoded)
                next_id += 1

        if len(existing_bytes) != len(self.vocab):
            raise ValueError("vocab contains duplicate byte representations")

        self.merges = list(merges)
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(piece, bytes) for piece in pair)
            for pair in self.merges
        ):
            raise TypeError("each merge must be a pair of bytes objects")

        self.special_tokens = normalized_specials
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in normalized_specials
        ]
        self.special_set = set(self.special_tokens_bytes)
        self._special_string_set = set(self.special_tokens)
        self.vocab_inv = {token: token_id for token_id, token in self.vocab.items()}

        rank: dict[tuple[int, int], int] = {}
        merge_to_new_id: dict[tuple[int, int], int] = {}
        for merge_rank, (left_bytes, right_bytes) in enumerate(self.merges):
            try:
                pair = (
                    self.vocab_inv[left_bytes],
                    self.vocab_inv[right_bytes],
                )
                new_id = self.vocab_inv[left_bytes + right_bytes]
            except KeyError as exc:
                raise ValueError(
                    f"merge rule {(left_bytes, right_bytes)!r} is not represented "
                    "in the vocabulary"
                ) from exc
            if pair in rank:
                raise ValueError(f"duplicate merge rule: {(left_bytes, right_bytes)!r}")
            rank[pair] = merge_rank
            merge_to_new_id[pair] = new_id

        self.rank = rank
        self.merge_to_new_id = merge_to_new_id
        self.eos_token_id = self.vocab_inv.get(b"<|endoftext|>")

    def _pre_tokenize(self, text: str) -> list[bytes]:
        """Return regex pretokens while preserving special tokens atomically."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        token_list: list[bytes] = []
        for part in split_by_special_tokens(
            text,
            self.special_tokens,
            include_special=True,
        ):
            if not part:
                continue
            if part in self._special_string_set:
                token_list.append(part.encode("utf-8"))
            else:
                token_list.extend(
                    match.group(0).encode("utf-8")
                    for match in re.finditer(PAT, part)
                )
        return token_list

    def _merge_one_pretoken(self, token_ids: list[int]) -> list[int]:
        """Apply ranked merge rules in O(n log n) time to one pretoken."""

        size = len(token_ids)
        if size < 2:
            return token_ids

        alive = [True] * size
        previous = [index - 1 for index in range(size)]
        following = [
            index + 1 if index + 1 < size else -1 for index in range(size)
        ]
        heap: list[tuple[int, int]] = []

        def push_pair(left_position: int) -> None:
            if left_position < 0 or not alive[left_position]:
                return
            right_position = following[left_position]
            if right_position == -1 or not alive[right_position]:
                return
            pair_rank = self.rank.get(
                (token_ids[left_position], token_ids[right_position])
            )
            if pair_rank is not None:
                heapq.heappush(heap, (pair_rank, left_position))

        for position in range(size - 1):
            push_pair(position)

        while heap:
            pair_rank, left_position = heapq.heappop(heap)
            right_position = following[left_position]
            if (
                right_position == -1
                or not alive[left_position]
                or not alive[right_position]
            ):
                continue

            pair = (
                token_ids[left_position],
                token_ids[right_position],
            )
            if self.rank.get(pair) != pair_rank:
                continue  # stale heap entry after a neighboring merge

            token_ids[left_position] = self.merge_to_new_id[pair]
            alive[right_position] = False
            next_position = following[right_position]
            following[left_position] = next_position
            if next_position != -1:
                previous[next_position] = left_position

            push_pair(previous[left_position])
            push_pair(left_position)

        result: list[int] = []
        position = 0
        while position != -1:
            if alive[position]:
                result.append(token_ids[position])
            position = following[position]
        return result

    def encode(self, text: str) -> list[int]:
        """Encode one string into token IDs."""

        encoded: list[int] = []
        for pretoken in self._pre_tokenize(text):
            if pretoken in self.special_set:
                encoded.append(self.vocab_inv[pretoken])
                continue
            byte_ids = list(pretoken)
            encoded.extend(self._merge_one_pretoken(byte_ids))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode strings from ``iterable`` without adding separators."""

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode IDs, replacing unknown IDs or invalid UTF-8 with U+FFFD."""

        pieces: list[bytes] = []
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            pieces.append(self.vocab.get(token_id, b"\xef\xbf\xbd"))
        return b"".join(pieces).decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: Sequence[str] | str | os.PathLike[str] | None = None,
    ) -> "BPETokenizer":
        """Load files written by :func:`save_vocab_and_merges`.

        GPT-2 byte-to-Unicode escaping is decoded back into the original bytes;
        this is essential for spaces, newlines, control bytes, and UTF-8 text.
        A special-token path ending in ``.json`` must contain a JSON string
        list; other paths retain support for the legacy one-token-per-line
        format.
        """

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
        if not isinstance(raw_vocab, dict):
            raise ValueError("vocab file must contain a JSON object")

        vocab: dict[int, bytes] = {}
        for text_token, token_id in raw_vocab.items():
            if not isinstance(text_token, str) or not isinstance(token_id, int):
                raise ValueError("vocab entries must map strings to integer IDs")
            if token_id in vocab:
                raise ValueError(f"duplicate token ID in vocab file: {token_id}")
            vocab[token_id] = text_to_token_bytes(text_token)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line_number, line in enumerate(merges_file, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"invalid merge rule on line {line_number}: {stripped!r}"
                    )
                merges.append(
                    (
                        text_to_token_bytes(parts[0]),
                        text_to_token_bytes(parts[1]),
                    )
                )

        if isinstance(special_tokens, (str, os.PathLike)):
            special_tokens_path = Path(special_tokens)
            with special_tokens_path.open(encoding="utf-8") as special_file:
                if special_tokens_path.suffix.casefold() == ".json":
                    raw_special_tokens = json.load(special_file)
                    if not isinstance(raw_special_tokens, list) or any(
                        not isinstance(token, str)
                        for token in raw_special_tokens
                    ):
                        raise ValueError(
                            "special-token JSON must contain a list of strings"
                        )
                    special_tokens_list = raw_special_tokens
                else:
                    special_tokens_list = [
                        line.rstrip("\r\n")
                        for line in special_file
                        if line.rstrip("\r\n")
                    ]
        elif special_tokens is None:
            special_tokens_list = []
        else:
            special_tokens_list = list(special_tokens)

        return cls(vocab, merges, special_tokens_list)


# The assignment-facing API commonly calls this class simply ``Tokenizer``.
# Keep BPETokenizer for backward compatibility and expose both names.
Tokenizer = BPETokenizer


def encode_file_to_bin(
    tokenizer: BPETokenizer,
    text_path: str | os.PathLike[str],
    out_bin_path: str | os.PathLike[str],
    dtype: Any = np.uint16,
) -> None:
    """Stream a UTF-8 text file into a flat NumPy token-ID binary file."""

    numpy_dtype = np.dtype(dtype)
    if numpy_dtype.kind != "u":
        raise ValueError("dtype must be an unsigned integer type")
    max_id = max(tokenizer.vocab, default=0)
    if max_id > np.iinfo(numpy_dtype).max:
        raise ValueError(
            f"dtype {numpy_dtype} cannot represent vocabulary ID {max_id}"
        )

    total_bytes = os.path.getsize(text_path)
    with (
        open(text_path, encoding="utf-8") as input_file,
        open(out_bin_path, "wb") as output_file,
        tqdm(
            total=total_bytes,
            desc="Encoding to binary",
            unit="B",
            unit_scale=True,
        ) as progress,
    ):
        for line in input_file:
            np.asarray(tokenizer.encode(line), dtype=numpy_dtype).tofile(output_file)
            progress.update(len(line.encode("utf-8")))


def load_tokenizer_from_dir(
    dir_path: str | os.PathLike[str],
) -> BPETokenizer:
    """Load a tokenizer directory produced by ``train_bpe(save_path=...)``."""

    directory = Path(dir_path)
    canonical_special_path = directory / "special_tokens.json"
    legacy_special_path = directory / "special_tokens.txt"
    if canonical_special_path.exists():
        special_path: Path | None = canonical_special_path
    elif legacy_special_path.exists():
        special_path = legacy_special_path
    else:
        special_path = None
    return BPETokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
        special_path,
    )
