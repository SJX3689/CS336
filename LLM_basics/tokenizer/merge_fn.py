"""Incremental data structures used by byte-pair encoding training."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass


Word = tuple[int, ...]
Pair = tuple[int, int]
Vocab = dict[int, bytes]
WordCounter = Counter[Word]
PairCounter = Counter[Pair]
PairToWords = dict[Pair, set[Word]]


def get_new_token_word(word: Word, target_pair: Pair, new_id: int) -> Word:
    """Merge all left-to-right, non-overlapping occurrences in one word.

    For example, merging ``(1, 1)`` in ``(1, 1, 1)`` produces
    ``(new_id, 1)``.  One input token cannot participate in two merges.
    """

    left_token, right_token = target_pair
    merged: list[int] = []
    position = 0
    while position < len(word):
        if (
            position + 1 < len(word)
            and word[position] == left_token
            and word[position + 1] == right_token
        ):
            merged.append(new_id)
            position += 2
        else:
            merged.append(word[position])
            position += 1
    return tuple(merged)


def need_merge(word: Word, target_pair: Pair) -> bool:
    """Return whether ``target_pair`` occurs adjacently in ``word``."""

    return any(pair == target_pair for pair in zip(word, word[1:]))


def _pairs_in_word(word: Word) -> Counter[Pair]:
    """Count adjacent pairs, including overlapping occurrences."""

    return Counter(zip(word, word[1:]))


def build_pair_counter(word_counter: dict[Word, int]) -> PairCounter:
    """Build corpus-wide adjacent-pair frequencies."""

    pair_counter: PairCounter = Counter()
    for word, frequency in word_counter.items():
        if frequency <= 0:
            continue
        for pair, occurrences in _pairs_in_word(word).items():
            pair_counter[pair] += frequency * occurrences
    return pair_counter


def build_pair_to_words(word_counter: dict[Word, int]) -> PairToWords:
    """Build the inverted index ``pair -> words containing that pair``."""

    pair_to_words: defaultdict[Pair, set[Word]] = defaultdict(set)
    for word, frequency in word_counter.items():
        if frequency <= 0:
            continue
        for pair in _pairs_in_word(word):
            pair_to_words[pair].add(word)
    return dict(pair_to_words)


@dataclass(slots=True)
class HeapItem:
    """One lazy-deletion heap record.

    The negative count makes the highest frequency sort first.  CS336's BPE
    convention resolves equal frequencies using the lexicographically greatest
    pair of byte strings, hence the reversed comparison in the tie case.
    """

    neg_freq: int
    pair_bytes: tuple[bytes, bytes]
    pair: Pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        if self.pair_bytes != other.pair_bytes:
            return self.pair_bytes > other.pair_bytes
        # This last branch only matters for malformed vocabularies containing
        # duplicate byte representations; it still gives heapq a total order.
        return self.pair < other.pair


def build_pair_heap(
    pair_counter: PairCounter,
    vocab: Vocab,
) -> list[HeapItem]:
    """Create a heap of all currently positive pair frequencies."""

    heap: list[HeapItem] = []
    for pair, frequency in pair_counter.items():
        if frequency <= 0:
            continue
        left_id, right_id = pair
        heapq.heappush(
            heap,
            HeapItem(
                -frequency,
                (vocab[left_id], vocab[right_id]),
                pair,
            ),
        )
    return heap


def pop_most_frequent_pair(
    pair_heap: list[HeapItem],
    pair_counter: PairCounter,
) -> Pair:
    """Return the best current pair after discarding stale heap records."""

    while pair_heap:
        item = pair_heap[0]
        current_frequency = pair_counter.get(item.pair, 0)
        if current_frequency <= 0 or -item.neg_freq != current_frequency:
            heapq.heappop(pair_heap)
            continue
        return item.pair
    raise ValueError("pair_heap is empty")


def merge_pairs_with_heap_index(
    word_counter: dict[Word, int],
    pair_counter: PairCounter,
    target_pair: Pair,
    new_id: int,
    vocab: Vocab,
    pair_heap: list[HeapItem],
    pair_to_words: PairToWords,
) -> tuple[WordCounter, PairCounter, list[HeapItem], PairToWords]:
    """Apply one BPE merge and update counters/indexes incrementally.

    ``vocab[new_id]`` must already exist because newly formed neighboring pairs
    are pushed into the heap using their byte representations.  Heap deletion
    is lazy: changed frequencies get fresh records and old records are removed
    by :func:`pop_most_frequent_pair` when they reach the top.
    """

    if new_id not in vocab:
        raise KeyError("new_id must be added to vocab before merging pairs")

    updated_words: WordCounter = Counter(word_counter)
    updated_pairs: PairCounter = pair_counter.copy()
    changed_pairs: set[Pair] = set()

    # Pair-count updates are commutative, so set iteration order cannot change
    # the result. Avoid sorting here because this is a hot path on large corpora.
    affected_words = tuple(pair_to_words.get(target_pair, set()))
    for old_word in affected_words:
        frequency = word_counter.get(old_word, 0)
        if frequency <= 0 or not need_merge(old_word, target_pair):
            continue

        del updated_words[old_word]
        old_pair_counts = _pairs_in_word(old_word)
        for old_pair, occurrences in old_pair_counts.items():
            updated_pairs[old_pair] -= frequency * occurrences
            if updated_pairs[old_pair] <= 0:
                updated_pairs.pop(old_pair, None)
            changed_pairs.add(old_pair)

            indexed_words = pair_to_words.get(old_pair)
            if indexed_words is not None:
                indexed_words.discard(old_word)
                if not indexed_words:
                    pair_to_words.pop(old_pair, None)

        new_word = get_new_token_word(old_word, target_pair, new_id)
        updated_words[new_word] += frequency
        for new_pair, occurrences in _pairs_in_word(new_word).items():
            updated_pairs[new_pair] += frequency * occurrences
            changed_pairs.add(new_pair)
            pair_to_words.setdefault(new_pair, set()).add(new_word)

    for pair in changed_pairs:
        frequency = updated_pairs.get(pair, 0)
        if frequency <= 0:
            continue
        left_id, right_id = pair
        heapq.heappush(
            pair_heap,
            HeapItem(
                -frequency,
                (vocab[left_id], vocab[right_id]),
                pair,
            ),
        )

    return updated_words, updated_pairs, pair_heap, pair_to_words
