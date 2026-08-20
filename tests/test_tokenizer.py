"""Regression tests for the byte-level BPE implementation.

These use only unittest so they run in the project's dependency-light virtual
environment, while remaining directly discoverable by pytest if it is added.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from LLM_basics.tokenizer import (
    BPETokenizer,
    Tokenizer,
    load_tokenizer_from_dir,
    train_bpe,
)
from LLM_basics.tokenizer.merge_fn import (
    build_pair_counter,
    build_pair_heap,
    build_pair_to_words,
    merge_pairs_with_heap_index,
    pop_most_frequent_pair,
)
from LLM_basics.tokenizer.tokenizer import (
    _count_corpus_pretokens,
    init_vocab,
    pre_tokenize,
    split_by_special_tokens,
)
from LLM_basics.tokenizer.utils import (
    text_to_token_bytes,
    token_bytes_to_text,
)


class BPETrainingTests(unittest.TestCase):
    def test_chunked_pretokenization_is_identical_to_whole_text(self) -> None:
        cases = [
            (
                "a  \n\n\t b\r\n   c\n\n",
                [],
            ),
            (
                "a  \n<short>\n\n b<long-special> c"
                "<long-special>\t\t<short>d",
                ["<short>", "<long-special>"],
            ),
            # The shorter token occurs inside the longer one. Selecting the
            # longest delimiter is necessary to avoid splitting that token.
            ("before axb after axb tail", ["x", "axb"]),
            # Raw occurrences overlap, so these cases exercise the conservative
            # single-range fallback rather than unsafe chunk boundaries.
            ("ababa " * 30, ["aba"]),
            ("xabc " * 30, ["xa", "abc"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (text, specials) in enumerate(cases):
                corpus = Path(directory) / f"corpus-{index}.txt"
                corpus.write_text(text, encoding="utf-8")
                chunked = _count_corpus_pretokens(corpus, specials, 8)
                self.assertEqual(chunked, pre_tokenize(text, specials))

    def test_training_is_deterministic_and_uses_required_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            # All three initial pairs occur once. Lexicographically, (a, c)
            # beats (a, b) and (space, a).
            corpus.write_text("ab ac", encoding="utf-8")

            first = train_bpe(corpus, 257, desired_num_chunks=1)
            second = train_bpe(corpus, 257, desired_num_chunks=1)

        self.assertEqual(first, second)
        vocab, merges = first
        self.assertEqual(merges, [(b"a", b"c")])
        self.assertEqual(vocab[256], b"ac")

    def test_empty_and_special_only_corpora_stop_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.txt"
            empty.write_text("", encoding="utf-8")
            empty_vocab, empty_merges = train_bpe(empty, 300)

            special_only = Path(directory) / "special.txt"
            special_only.write_text("<eos><eos>", encoding="utf-8")
            special_vocab, special_merges = train_bpe(
                special_only,
                300,
                ["<eos>"],
            )

        self.assertEqual(len(empty_vocab), 256)
        self.assertEqual(empty_merges, [])
        self.assertEqual(len(special_vocab), 257)
        self.assertEqual(special_vocab[256], b"<eos>")
        self.assertEqual(special_merges, [])

    def test_incremental_pair_index_matches_full_recomputation(self) -> None:
        random_source = random.Random(336)
        words: Counter[tuple[int, ...]] = Counter()
        for _ in range(40):
            word = tuple(
                random_source.randrange(6)
                for _ in range(random_source.randrange(1, 10))
            )
            words[word] += random_source.randrange(1, 5)

        vocab = {token_id: bytes((65 + token_id,)) for token_id in range(6)}
        pairs = build_pair_counter(words)
        pair_to_words = build_pair_to_words(words)
        heap = build_pair_heap(pairs, vocab)

        for _ in range(12):
            if not pairs:
                break
            expected = max(
                pairs,
                key=lambda pair: (
                    pairs[pair],
                    (vocab[pair[0]], vocab[pair[1]]),
                ),
            )
            selected = pop_most_frequent_pair(heap, pairs)
            self.assertEqual(selected, expected)

            new_id = max(vocab) + 1
            vocab[new_id] = vocab[selected[0]] + vocab[selected[1]]
            words, pairs, heap, pair_to_words = merge_pairs_with_heap_index(
                words,
                pairs,
                selected,
                new_id,
                vocab,
                heap,
                pair_to_words,
            )
            self.assertEqual(pairs, build_pair_counter(words))
            self.assertEqual(pair_to_words, build_pair_to_words(words))

    def test_argument_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text("text", encoding="utf-8")
            with self.assertRaises(ValueError):
                train_bpe(corpus, 255)
            with self.assertRaises(ValueError):
                train_bpe(corpus, 256, [""])
            with self.assertRaises(ValueError):
                train_bpe(corpus, 256, ["<x>", "<x>"])


class TokenizerTests(unittest.TestCase):
    def _trained_tokenizer(self, directory: str) -> tuple[Tokenizer, Path]:
        corpus = Path(directory) / "corpus.txt"
        corpus.write_text(
            "low low lower\n你好，世界！<|endoftext|>low\n",
            encoding="utf-8",
        )
        save_path = Path(directory) / "tokenizer"
        vocab, merges = train_bpe(
            corpus,
            290,
            ["<|endoftext|>"],
            desired_num_chunks=3,
            save_path=save_path,
        )
        return Tokenizer(vocab, merges, ["<|endoftext|>"]), save_path

    def test_unicode_special_and_iterable_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokenizer, _ = self._trained_tokenizer(directory)
            text = "low 你好🙂\n<|endoftext|> unseen"
            token_ids = tokenizer.encode(text)

            self.assertEqual(tokenizer.decode(token_ids), text)
            self.assertIn(tokenizer.eos_token_id, token_ids)
            self.assertEqual(
                list(tokenizer.encode_iterable(["low", "🙂"])),
                tokenizer.encode("low") + tokenizer.encode("🙂"),
            )

    def test_save_load_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokenizer, save_path = self._trained_tokenizer(directory)
            loaded = load_tokenizer_from_dir(save_path)
            legacy_loaded = BPETokenizer.from_files(
                save_path / "vocab.json",
                save_path / "merges.txt",
                save_path / "special_tokens.txt",
            )
            sample = " space\n\t中文🙂<|endoftext|>"

            self.assertTrue((save_path / "special_tokens.json").exists())
            self.assertTrue((save_path / "special_tokens.txt").exists())
            self.assertEqual(loaded.vocab, tokenizer.vocab)
            self.assertEqual(loaded.merges, tokenizer.merges)
            self.assertEqual(loaded.encode(sample), tokenizer.encode(sample))
            self.assertEqual(loaded.decode(loaded.encode(sample)), sample)
            self.assertEqual(legacy_loaded.encode(sample), tokenizer.encode(sample))

    def test_newline_special_tokens_survive_save_and_load(self) -> None:
        specials = ["<line\nbreak>", "<windows\r\nline>"]
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text(
                f"before{specials[0]}middle{specials[1]}after",
                encoding="utf-8",
            )
            save_path = Path(directory) / "tokenizer"
            vocab, merges = train_bpe(
                corpus,
                256 + len(specials),
                specials,
                desired_num_chunks=2,
                save_path=save_path,
            )

            canonical_path = save_path / "special_tokens.json"
            self.assertEqual(
                json.loads(canonical_path.read_text(encoding="utf-8")),
                specials,
            )
            # The legacy line format would corrupt these tokens, so it must not
            # be emitted as an incomplete or ambiguous fallback.
            legacy_path = save_path / "special_tokens.txt"
            self.assertFalse(legacy_path.exists())

            # Even if an old or externally supplied legacy file is present,
            # directory loading treats the lossless JSON file as authoritative.
            legacy_path.write_text("<stale-token>\n", encoding="utf-8")

            loaded = load_tokenizer_from_dir(save_path)
            loaded_directly = BPETokenizer.from_files(
                save_path / "vocab.json",
                save_path / "merges.txt",
                canonical_path,
            )
            sample = f"x{specials[0]}y{specials[1]}z"
            expected_ids = BPETokenizer(vocab, merges, specials).encode(sample)

            self.assertEqual(loaded.special_tokens, specials)
            self.assertEqual(loaded.encode(sample), expected_ids)
            self.assertEqual(loaded_directly.encode(sample), expected_ids)
            self.assertEqual(loaded.decode(expected_ids), sample)

    def test_constructor_adds_missing_runtime_special(self) -> None:
        base_vocab = init_vocab()
        tokenizer = BPETokenizer(base_vocab, [], ["<runtime>"])

        self.assertNotIn(b"<runtime>", base_vocab.values())
        self.assertEqual(tokenizer.encode("<runtime>"), [256])
        self.assertEqual(tokenizer.decode([256]), "<runtime>")

    def test_unknown_id_decodes_to_replacement_character(self) -> None:
        tokenizer = Tokenizer(init_vocab(), [])
        self.assertEqual(tokenizer.decode([999_999]), "\ufffd")

    def test_special_splitting_and_pretoken_counts(self) -> None:
        pieces = split_by_special_tokens(
            "a<xy>b<x>c",
            ["<x>", "<xy>"],
            include_special=True,
        )
        self.assertEqual(pieces, ["a", "<xy>", "b", "<x>", "c"])
        counts = pre_tokenize("a<eos>a", ["<eos>"])
        self.assertEqual(counts, Counter({(97,): 2}))

    def test_byte_unicode_file_alphabet_is_reversible(self) -> None:
        every_byte = bytes(range(256))
        self.assertEqual(
            text_to_token_bytes(token_bytes_to_text(every_byte)),
            every_byte,
        )


if __name__ == "__main__":
    unittest.main()
