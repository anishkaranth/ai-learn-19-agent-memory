"""Smoke run: 40 seeded 80-turn conversations x 5 memory strategies.
Writes results/metrics.json, results/JSON.shot, results/RESULTS.md and SVG plots."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from conversation import SLOTS, make_episodes
from evaluate import BINS, run
from memory import FullHistory, HybridMemory, RollingSummary, VectorMemory, WindowBuffer
from smoke_plots import make_plots, write_results_md

SEED = 42
CONFIG = {"episodes": 40, "turns": 80, "facts_per_episode": len(SLOTS),
          "window": {"exchanges": 6}, "vector": {"k": 3, "dim": 512, "embedder": "hashed bag-of-stems"},
          "summary": {"block": 8, "keep": 2, "budget_words": 60},
          "hybrid": {"window": 4, "k": 2, "block": 8, "keep": 2, "budget_words": 40},
          "token_unit": "whitespace word", "distance_bins": [f"{a}-{b}" for a, b in BINS]}


def factories():
    c = CONFIG
    return {
        "full_history": FullHistory,
        "window": lambda: WindowBuffer(c["window"]["exchanges"]),
        "vector": lambda: VectorMemory(c["vector"]["k"], c["vector"]["dim"]),
        "summary": lambda: RollingSummary(c["summary"]["block"], c["summary"]["keep"], c["summary"]["budget_words"]),
        "hybrid": lambda: HybridMemory(c["hybrid"]["window"], c["hybrid"]["k"], c["hybrid"]["block"], c["hybrid"]["keep"], c["hybrid"]["budget_words"]),
    }


def main() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.perf_counter()
    episodes = make_episodes(CONFIG["episodes"], CONFIG["turns"], SEED)
    res = run(episodes, factories())
    m = {"project": "ai-learn-19-agent-memory", "seed": SEED, "config": CONFIG, "results": res}
    m["runtime_s"] = time.perf_counter() - t0
    out = Path(__file__).parent / "results"
    plots = make_plots(out, m)
    write_results_md(out, m, plots)
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    shot = {"project": m["project"], "seed": SEED,
            "config": {k: CONFIG[k] for k in ("episodes", "turns", "facts_per_episode", "window", "vector", "summary", "hybrid")},
            "summary": {s: {"contain": round(r["contain"], 4), "reader_acc": round(r["reader_acc"], 4),
                            "ctx_tokens_mean": round(r["ctx_tokens_mean"], 1), "ctx_tokens_turn_peak": round(r["ctx_tokens_turn_peak"], 1),
                            "contain_by_distance": {b: round(v["contain"], 4) for b, v in r["by_distance"].items()}}
                        for s, r in res.items()},
            "plots": plots, "runtime_s": round(m["runtime_s"], 3)}
    (out / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    return m


if __name__ == "__main__":
    m = main()
    for s, r in m["results"].items():
        print(f"{s:13s} contain={r['contain']:.3f} reader={r['reader_acc']:.3f} tokens={r['ctx_tokens_mean']:.1f} "
              + " ".join(f"{b}:{v['contain']:.2f}" for b, v in r["by_distance"].items()))
    print(f"runtime {m['runtime_s']:.2f}s")
