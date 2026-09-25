"""Memory strategies for a toy conversational agent. Each strategy observes turns one at a time and,
when asked a question, returns the list of context lines it would put in the prompt."""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import List

import numpy as np

from text import tokenize


def n_tokens(lines: List[str]) -> int:
    """Token proxy: whitespace words."""
    return sum(len(l.split()) for l in lines)


class FullHistory:
    """Upper bound: keep every exchange (cost grows linearly with the conversation)."""
    name = "full_history"

    def __init__(self):
        self.lines: List[str] = []

    def observe(self, turn):
        self.lines += [f"User: {turn.user}", f"Agent: {turn.agent}"]

    def context(self, question: str) -> List[str]:
        return list(self.lines)


class WindowBuffer:
    """Short-term memory: the last `window` exchanges only."""
    name = "window"

    def __init__(self, window: int = 6):
        self.buf = deque(maxlen=window)

    def observe(self, turn):
        self.buf.append((f"User: {turn.user}", f"Agent: {turn.agent}"))

    def context(self, question: str) -> List[str]:
        return [l for pair in self.buf for l in pair]


class HashEmbedder:
    """Bag-of-stems hashed into `dim` buckets, L2-normalised (a stand-in for a sentence encoder)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for t in set(tokenize(text)):
            v[hash_token(t) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


def hash_token(t: str) -> int:
    h = 2166136261  # FNV-1a, deterministic across runs (unlike Python's salted hash())
    for ch in t.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


class VectorMemory:
    """Long-term memory: embed every user utterance, retrieve the top-k most similar to the question."""
    name = "vector"

    def __init__(self, k: int = 3, dim: int = 512):
        self.k, self.embed = k, HashEmbedder(dim)
        self.texts: List[str] = []
        self.vecs: List[np.ndarray] = []

    def observe(self, turn):
        self.texts.append(f"User: {turn.user}")
        self.vecs.append(self.embed(turn.user))

    def retrieve(self, question: str, k: int) -> List[str]:
        if not self.texts:
            return []
        sims = np.stack(self.vecs) @ self.embed(question)
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.texts[i] for i in sorted(order)]  # keep chronological order in the prompt

    def context(self, question: str) -> List[str]:
        return self.retrieve(question, self.k)


class RollingSummary:
    """Extractive rolling summary: every `block` turns, keep the `keep` most salient user sentences
    (mean IDF of their stems, IDF computed online over the conversation). When the summary exceeds
    `budget` words, drop the least salient lines (re-scored with the current IDF). Context = summary + raw
    turns not yet summarised."""
    name = "summary"

    def __init__(self, block: int = 8, keep: int = 2, budget: int = 60):
        self.block, self.keep, self.budget = block, keep, budget
        self.df: Counter = Counter()
        self.n_docs = 0
        self.pending: List = []
        self.summary: List[str] = []

    def _salience(self, sent: str) -> float:
        toks = set(tokenize(sent))
        if not toks:
            return 0.0
        return sum(math.log((1 + self.n_docs) / (1 + self.df[t])) + 1 for t in toks) / len(toks)

    def observe(self, turn):
        self.n_docs += 1
        self.df.update(set(tokenize(turn.user)))
        self.pending.append(turn)
        if len(self.pending) >= self.block:
            ranked = sorted(self.pending, key=lambda t: -self._salience(t.user))[: self.keep]
            self.summary += [f"Summary: {t.user}" for t in sorted(ranked, key=lambda t: t.idx)]
            self.pending = []
            while n_tokens(self.summary) > self.budget:
                worst = min(range(len(self.summary)), key=lambda i: self._salience(self.summary[i]))
                self.summary.pop(worst)

    def context(self, question: str) -> List[str]:
        return self.summary + [l for t in self.pending for l in (f"User: {t.user}", f"Agent: {t.agent}")]


class HybridMemory:
    """Window buffer + rolling summary + vector recall, de-duplicated."""
    name = "hybrid"

    def __init__(self, window: int = 4, k: int = 2, block: int = 8, keep: int = 2, budget: int = 40):
        self.win, self.vec = WindowBuffer(window), VectorMemory(k)
        self.summ = RollingSummary(block, keep, budget)

    def observe(self, turn):
        self.win.observe(turn); self.vec.observe(turn); self.summ.observe(turn)

    def context(self, question: str) -> List[str]:
        out, seen = [], set()
        for l in self.summ.summary + self.vec.context(question) + self.win.context(question):
            key = l.split(": ", 1)[1]
            if l.startswith("Agent:") or key not in seen:
                out.append(l)
                seen.add(key)
        return out


def extractive_reader(question: str, context: List[str]) -> str:
    """Toy reader: pick the user/summary line with the most stem overlap with the question
    (ties -> most recent). Returns that line; correctness = gold value appears in it."""
    q = set(tokenize(question))
    best, best_s = "", 0
    for line in context:
        if line.startswith("Agent:"):
            continue
        s = len(q & set(tokenize(line)))
        if s >= best_s and s > 0:
            best, best_s = line, s
    return best
