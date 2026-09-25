"""Replay conversations through every memory strategy and score fact recall at the end."""
from __future__ import annotations

import re
from typing import Callable, Dict, List

import numpy as np

from memory import extractive_reader, n_tokens

BINS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 80)]
REF_QUESTION = "What is my dog called?"  # used to track the context size turn by turn


def mentions(line: str, value: str) -> bool:
    """Whole-word match, so 'pho' does not match inside 'phone'."""
    return re.search(r"\b" + re.escape(value) + r"\b", line) is not None


def run(episodes, factories: Dict[str, Callable]) -> Dict[str, dict]:
    n_turns = len(episodes[0][0])
    out = {}
    for name, make in factories.items():
        contain, correct, dist, tokens = [], [], [], []
        curve = np.zeros(n_turns)
        for turns, probes in episodes:
            m = make()
            for t in turns:
                m.observe(t)
                curve[t.idx] += n_tokens(m.context(REF_QUESTION))
            for p in probes:
                ctx = m.context(p.question)
                contain.append(any(mentions(l, p.answer) for l in ctx))
                correct.append(mentions(extractive_reader(p.question, ctx), p.answer))
                dist.append(n_turns - p.fact_turn)
                tokens.append(n_tokens(ctx))
        c, r, d = np.array(contain), np.array(correct), np.array(dist)
        by_bin = {}
        for a, b in BINS:
            mask = (d >= a) & (d <= b)
            by_bin[f"{a}-{b}"] = {"n": int(mask.sum()), "contain": float(c[mask].mean()), "reader_acc": float(r[mask].mean())}
        out[name] = {"contain": float(c.mean()), "reader_acc": float(r.mean()), "n_probes": int(len(c)),
                     "ctx_tokens_mean": float(np.mean(tokens)), "ctx_tokens_max": int(np.max(tokens)),
                     "reader_acc_per_100_tokens": float(r.mean() / np.mean(tokens) * 100),
                     "ctx_tokens_turn_mean": float(curve.mean() / len(episodes)), "ctx_tokens_turn_peak": float(curve.max() / len(episodes)),
                     "by_distance": by_bin, "ctx_tokens_by_turn": [float(x) for x in curve / len(episodes)]}
    return out
