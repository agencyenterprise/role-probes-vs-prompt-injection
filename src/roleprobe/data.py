"""Corpus loading and role-wrapped training sequences."""
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from .models import ROLES


def _c4(seed):
    return load_dataset('allenai/c4', 'en', split='validation',
                        streaming=True).shuffle(seed=seed, buffer_size=50_000)


def _dolma3(seed):
    return load_dataset('allenai/dolma3_mix-150B-1025', split='train', revision='3a8349c',
                        streaming=True).shuffle(seed=seed, buffer_size=50_000)


CORPORA = {'c4': _c4, 'dolma3': _dolma3}


def parse_mix(spec):
    """'c4=0.25,dolma3=0.75' -> {'c4': 0.25, 'dolma3': 0.75}; bare names weigh 1."""
    parts = [p.partition('=') for p in spec.split(',')]
    return {name: float(w) if w else 1.0 for name, _, w in parts}


def load_texts(mix, n_docs, seed):
    weights = np.array(list(mix.values()), dtype=float)
    counts = np.floor(weights / weights.sum() * n_docs).astype(int)
    counts[0] += n_docs - counts.sum()
    texts = []
    for name, n in zip(mix, counts):
        stream = iter(CORPORA[name](seed))
        texts += [next(stream)['text'] for _ in range(n)]
    return texts


@dataclass(frozen=True)
class Sequence:
    doc_ix: int
    role: str
    ids: tuple
    content: slice  # content-token positions within ids


def build_sequences(texts, tokenizer, fmt, seq_len, tool_names=('',), seed=0):
    """Render each document's first seq_len tokens as every role.

    Wrappers and content are tokenized separately, so every token's role label
    is exact by construction — no post-hoc string matching.
    """
    rng = np.random.default_rng(seed)
    ids = lambda s: tuple(tokenizer(s, add_special_tokens=False).input_ids)
    docs = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=seq_len).input_ids

    seqs = []
    for doc_ix, content in enumerate(docs):
        tool = rng.choice(tool_names)
        for role in ROLES:
            prefix, suffix = fmt.wrap[role]
            pre, post = ids(prefix.format(tool_name=tool)), ids(suffix)
            seqs.append(Sequence(doc_ix, role, pre + tuple(content) + post,
                                 slice(len(pre), len(pre) + len(content))))
    return seqs
