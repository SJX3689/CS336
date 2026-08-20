import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import BinaryIO, Callable, ParamSpec, Sequence, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    mini_chunk_size: int = 4096,
) -> list[int]:
    """
    Divide a binary file into approximately equal chunks.

    Internal boundaries are moved forward to the beginning of the next
    occurrence of `split_special_token`, so the special token is not split
    between chunks.

    Returns byte offsets using left-closed, right-open intervals:
        [boundary[i], boundary[i + 1])

    The returned number of chunks may be smaller than desired_num_chunks
    when multiple estimated boundaries locate the same special token.
    """
    if isinstance(desired_num_chunks, bool) or not isinstance(
        desired_num_chunks, int
    ):
        raise TypeError("desired_num_chunks must be an integer")
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be bytes")
    if not split_special_token:
        raise ValueError("split_special_token must not be empty")
    if mini_chunk_size <= 0:
        raise ValueError("mini_chunk_size must be positive")

    original_position = file.tell()

    try:
        # Obtain file size in bytes.
        file.seek(0, os.SEEK_END)
        file_size = file.tell()

        if file_size == 0:
            return [0]

        # Avoid creating many duplicate initial boundaries for very small files.
        num_chunks = min(desired_num_chunks, file_size)
        chunk_size = file_size // num_chunks

        chunk_boundaries = [
            i * chunk_size for i in range(num_chunks + 1)
        ]
        chunk_boundaries[-1] = file_size

        overlap_size = len(split_special_token) - 1

        # Adjust only internal boundaries.
        for boundary_index in range(1, len(chunk_boundaries) - 1):
            search_start = chunk_boundaries[boundary_index]
            file.seek(search_start)

            current_position = search_start
            overlap = b""

            while True:
                data = file.read(mini_chunk_size)

                if not data:
                    chunk_boundaries[boundary_index] = file_size
                    break

                # Retain bytes from the previous block so a special token
                # spanning two reads can still be detected.
                search_data = overlap + data
                found_at = search_data.find(split_special_token)

                if found_at != -1:
                    absolute_position = (
                        current_position - len(overlap) + found_at
                    )
                    chunk_boundaries[boundary_index] = absolute_position
                    break

                if overlap_size > 0:
                    overlap = search_data[-overlap_size:]
                else:
                    overlap = b""

                current_position += len(data)

        return sorted(set(chunk_boundaries))

    finally:
        # Do not unexpectedly change the caller's file position.
        file.seek(original_position)


def string_to_bytes(
    text: str,
    return_int: bool = False,
) -> list[int] | list[bytes]:
    """
    Encode a string as UTF-8.

    When return_int=True:
        "abc" -> [97, 98, 99]

    When return_int=False:
        "abc" -> [b"a", b"b", b"c"]
    """
    encoded = text.encode("utf-8")

    if return_int:
        return list(encoded)

    return [bytes((byte,)) for byte in encoded]


def utf8_bytes_to_string(byte_tokens: Sequence[bytes]) -> str:
    """
    Join byte tokens and decode the result as UTF-8.
    """
    return b"".join(byte_tokens).decode("utf-8")


def bytes_to_unicode() -> dict[int, str]:
    """
    Construct the reversible byte-to-Unicode mapping used by GPT-2-style
    byte-level BPE.

    Every byte from 0 to 255 is mapped to one printable Unicode character.
    This avoids writing raw spaces, newlines and control bytes directly into
    vocab.json or merges.txt.
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    unicode_values = byte_values.copy()
    extra_index = 0

    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_values.append(256 + extra_index)
            extra_index += 1

    return {
        byte_value: chr(unicode_value)
        for byte_value, unicode_value in zip(
            byte_values,
            unicode_values,
        )
    }


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {character: byte for byte, character in _BYTE_ENCODER.items()}


def token_bytes_to_text(token: bytes) -> str:
    """
    Convert arbitrary bytes to a reversible, text-safe representation.
    """
    return "".join(_BYTE_ENCODER[byte] for byte in token)


def text_to_token_bytes(text: str) -> bytes:
    """Invert :func:`token_bytes_to_text`.

    ``vocab.json`` and ``merges.txt`` use printable Unicode characters rather
    than raw control bytes.  Decoding with Latin-1 (as the previous loader did)
    is not the inverse of that representation: for example byte 32 is stored as
    ``Ġ``.  Rejecting unknown characters also makes malformed files fail early.
    """

    try:
        return bytes(_BYTE_DECODER[character] for character in text)
    except KeyError as exc:
        raise ValueError(
            f"token contains a character outside the byte-level alphabet: {exc.args[0]!r}"
        ) from exc


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: Sequence[tuple[bytes, bytes]],
    output_dir: str | os.PathLike[str],
) -> None:
    """
    Save a byte-level BPE vocabulary and merge rules.

    vocab:
        Mapping from token ID to token bytes.

    merges:
        Ordered BPE merge rules. Earlier entries have higher priority.
    """
    if len(vocab) != len(set(vocab.values())):
        raise ValueError("vocab contains duplicate byte tokens")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in vocab
    ):
        raise TypeError("vocab IDs must be integers")
    if any(not isinstance(token, bytes) for token in vocab.values()):
        raise TypeError("vocab values must be bytes")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_filepath = output_path / "vocab.json"
    merges_filepath = output_path / "merges.txt"

    # Convert token_id -> bytes into text_token -> token_id.
    vocab_inverse = {
        token_bytes_to_text(token): token_id
        for token_id, token in sorted(vocab.items())
    }

    with vocab_filepath.open("w", encoding="utf-8") as vocab_file:
        json.dump(
            vocab_inverse,
            vocab_file,
            ensure_ascii=False,
            indent=2,
        )
        vocab_file.write("\n")

    with merges_filepath.open("w", encoding="utf-8") as merges_file:
        merges_file.write("#version: 0.2\n")

        for left_token, right_token in merges:
            left_text = token_bytes_to_text(left_token)
            right_text = token_bytes_to_text(right_token)
            merges_file.write(f"{left_text} {right_text}\n")


def timeit(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that prints a function's execution time.

    Timing is printed even if the wrapped function raises an exception.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[TIME] {func.__name__} took {elapsed:.2f}s")

    return wrapper

def print_color(content: str, color: str = "green") -> None:
    """Print a small Rich-style color tag without requiring Rich itself."""

    print(f"[{color}]{content}[/{color}]")
