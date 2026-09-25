"""Matplotlib SVG plots + RESULTS.md writer for the agent-memory smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ai-learn-19"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_META = {"Date": None}
_C = {"full_history": "#adb5bd", "window": "#264653", "vector": "#2a9d8f", "summary": "#e9c46a", "hybrid": "#e76f51"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata=_META)
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    out.mkdir(exist_ok=True)
    res, names = m["results"], []
    bins = m["config"]["distance_bins"]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for s, r in res.items():
        ax.plot(range(len(bins)), [r["by_distance"][b]["contain"] for b in bins], "o-", ms=4, color=_C[s], label=s)
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlabel("fact stated N turns before the question")
    ax.set_ylabel("fact in context")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall of a fact vs how long ago it was stated")
    ax.legend(fontsize=8, loc="lower left")
    names.append(_save(fig, out / "recall_vs_distance.svg"))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(res))
    ax.bar(x - 0.2, [r["contain"] for r in res.values()], 0.4, color="#2a9d8f", label="fact in context")
    ax.bar(x + 0.2, [r["reader_acc"] for r in res.values()], 0.4, color="#e9c46a", label="extractive reader correct")
    ax.set_xticks(x, list(res))
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall recall vs reader accuracy (all distances)")
    ax.legend(fontsize=8, loc="upper right")
    names.append(_save(fig, out / "recall_vs_reader.svg"))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for s, r in res.items():
        ax.plot(np.arange(1, len(r["ctx_tokens_by_turn"]) + 1), r["ctx_tokens_by_turn"], color=_C[s], label=s, lw=1.5)
    ax.set_xlabel("turn")
    ax.set_ylabel("context words")
    ax.set_title("Token budget: prompt context size as the conversation grows")
    ax.legend(fontsize=8, loc="upper left")
    names.append(_save(fig, out / "token_budget.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c, res = m["config"], m["results"]
    bins = c["distance_bins"]
    L = ["# Results -- ai-learn-19-agent-memory", "",
         f"**Seed:** `{m['seed']}` | {c['episodes']} conversations x {c['turns']} turns | {c['facts_per_episode']} facts per conversation, "
         f"probed at the end ({next(iter(res.values()))['n_probes']} probes per strategy) | tokens = whitespace words", "",
         "`contain`: the gold value appears somewhere in the context the memory returns. `reader`: a toy extractive reader "
         "(best stem overlap with the question) picks a line containing the value.", "",
         "## Overall (real smoke run)", "",
         "| strategy | contain | reader acc | ctx words at probe (mean) | ctx words at probe (max) | ctx words per turn (mean / peak) | reader acc / 100 words |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s, r in res.items():
        L.append(f"| `{s}` | {r['contain']:.3f} | {r['reader_acc']:.3f} | {r['ctx_tokens_mean']:.1f} | {r['ctx_tokens_max']} | {r['ctx_tokens_turn_mean']:.1f} / {r['ctx_tokens_turn_peak']:.1f} | {r['reader_acc_per_100_tokens']:.2f} |")
    for key, title in (("contain", "Fact in context"), ("reader_acc", "Reader accuracy")):
        L += ["", f"## {title} by distance N (turns between statement and question)", "",
              "| strategy | " + " | ".join(f"N={b}" for b in bins) + " |", "|---|" + "---:|" * len(bins)]
        for s, r in res.items():
            L.append(f"| `{s}` | " + " | ".join(f"{r['by_distance'][b][key]:.3f}" for b in bins) + " |")
    L += ["", "Probes per distance bin: " + ", ".join(f"N={b}: {v['n']}" for b, v in next(iter(res.values()))["by_distance"].items()), "",
          "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['runtime_s']:.2f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
