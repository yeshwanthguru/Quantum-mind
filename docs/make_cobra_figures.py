"""Draw the COBRA figures used in the README and the documentation (python docs/make_cobra_figures.py).

Trains one COBRA and one Gaussian Bayes filter on the synthetic *context* intent task of
``benchmarks/context_intent_benchmark.py`` and writes PNGs to ``docs/assets/``:

  cobra_context_effect.png    belief shift from one hand cue after opposite previous looks
  cobra_belief_timeline.png   sensor cues and belief per frame, one episode per intent
  cobra_density_matrix.png    |rho| after each frame (diagonal = probabilities, off-diagonal = coherence)
  cobra_benchmark.png         expected cost against training size, from the benchmark results

Needs numpy and matplotlib. ``--no-train`` redraws only the benchmark chart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from context_intent_benchmark import D, M, T, oracle_proba, simulate  # noqa: E402
from quantum_mind.belief import COBRA, GaussianBayesFilter  # noqa: E402

OUT = ROOT / "docs" / "assets"
INTENTS = ["handover", "retrieve", "withdraw"]
CUES = ["gaze cue a", "hand cue b", "force cue f"]
C_INT = ["#0072B2", "#E69F00", "#CC79A7"]          # intent colours (colour-blind safe)
C_CUE = ["#009E73", "#56B4E9", "#D55E00"]
MODEL_STYLE = {"Oracle": ("#222222", "--"), "COBRA": ("#0072B2", "-"),
               "COBRA (no coherence)": ("#56B4E9", "-"), "Bayes filter": ("#D55E00", "-"),
               "Logistic reg.": ("#999999", "-"), "MLP (32 hidden)": ("#CC79A7", "-")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def train_models(seed: int = 0):
    rng = np.random.default_rng(999)
    x, z = simulate("context", 1000, rng)
    cobra = COBRA(D, M, seed=seed, restarts=8, batch=100).fit(x, z)
    bayes = GaussianBayesFilter(D).fit(x, z)
    return cobra, bayes


def pick_episodes(cobra, rng):
    """One correctly-classified test episode per intent (shows typical behaviour)."""
    x, z = simulate("context", 600, rng)
    pred = cobra.predict_proba(x).argmax(1)
    idx = [int(np.flatnonzero((z == k) & (pred == k))[0]) for k in range(D)]
    return x[idx], z[idx]


def fig_timeline(cobra, eps, labels):
    tr = cobra.trace(eps)
    frames = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, D, figsize=(11, 5.2), sharex=True, sharey="row")
    for k in range(D):
        for c in range(M):
            ax[0, k].plot(frames, eps[k, :, c], "o-", color=C_CUE[c], label=CUES[c])
        ax[0, k].axhline(0, color="k", lw=0.6)
        ax[0, k].set_title(f"True intent: {INTENTS[labels[k]]}")
        for h in range(D):
            ax[1, k].plot(frames, tr.probs[k, :, h], "o-", color=C_INT[h], label=INTENTS[h],
                          lw=2.4 if h == labels[k] else 1.4)
        ax[1, k].set_ylim(0, 1.02)
        ax[1, k].set_xlabel("frame")
        ax[1, k].set_xticks(frames)
    ax[0, 0].set_ylabel("sensor value")
    ax[1, 0].set_ylabel("belief P(intent)")
    ax[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    ax[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("COBRA belief over time for one test episode per intent (thick line = true intent)", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "cobra_belief_timeline.png")
    plt.close(fig)


def fig_density(cobra, eps, labels):
    tr = cobra.trace(eps)
    k = 0
    fig, ax = plt.subplots(1, T, figsize=(11, 3.2))
    for t in range(T):
        im = ax[t].imshow(np.abs(tr.rho[k, t]), vmin=0, vmax=1, cmap="viridis")
        ax[t].set_title(f"after frame {t + 1}\ncoherence {tr.coherence[k, t]:.2f}", fontsize=9)
        ax[t].set_xticks(range(D), [n[:4] for n in INTENTS], fontsize=8)
        ax[t].set_yticks(range(D), [n[:4] for n in INTENTS] if t == 0 else [], fontsize=8)
        ax[t].grid(False)
        for i in range(D):
            for j in range(D):
                v = abs(tr.rho[k, t, i, j])
                ax[t].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                           color="w" if v < 0.55 else "k")
    fig.colorbar(im, ax=ax, shrink=0.8, label="|rho|")
    fig.suptitle(f"Density matrix |rho| for a '{INTENTS[labels[k]]}' episode (diagonal = probabilities, off-diagonal = coherence)",
                 y=1.02)
    fig.savefig(OUT / "cobra_density_matrix.png", bbox_inches="tight")
    plt.close(fig)


def fig_context(cobra, bayes):
    """Belief shift caused by one hand cue b=+1.5, after opposite previous gaze cues.

    The previous gaze cue is informative on its own, so the figure plots only the
    *change* in log-odds(handover : retrieve) that the hand-cue frame adds. A
    commutative filter adds the same amount whatever came before; a model that
    uses context adds an amount whose sign depends on the previous gaze cue.
    """
    def logodds(p):
        return float(np.log(p[0] / p[1]))

    def shift(fn, sign):
        before = np.zeros((1, 1, M)); before[0, 0, 0] = sign * 1.5
        after = np.zeros((1, 2, M)); after[0, 0, 0] = sign * 1.5; after[0, 1, 1] = 1.5
        return logodds(fn(after)[0]) - logodds(fn(before)[0])

    models = [("COBRA", cobra.predict_proba), ("Bayes filter", bayes.predict_proba),
              ("Oracle", lambda s: oracle_proba("context", s))]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.36
    for i, sign in enumerate((+1, -1)):
        vals = [shift(fn, sign) for _, fn in models]
        bars = ax.bar(np.arange(len(models)) + (i - 0.5) * w, vals, w,
                      color=["#0072B2", "#E69F00"][i], label=f"previous gaze cue a = {sign:+.1f}")
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + (0.15 if v >= 0 else -0.15), f"{v:+.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(models)), [n for n, _ in models])
    ax.set_ylabel("shift in log-odds(handover : retrieve)\ncaused by hand cue b = +1.5")
    ax.set_ylim(-5.5, 5.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Does the same hand cue mean different things after different gaze cues?", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "cobra_context_effect.png")
    plt.close(fig)


def fig_benchmark():
    path = ROOT / "benchmarks" / "results" / "context_intent" / "benchmark_raw.json"
    rows = json.loads(path.read_text())
    panels = [("context", "clean", "Context task"), ("context", "noise x1.5", "Context task, 1.5x test noise"),
              ("control", "clean", "Control task (no context effect)")]
    sizes = sorted({r["n"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for a, (kind, cond, title) in zip(ax, panels):
        for name, (col, ls) in MODEL_STYLE.items():
            mean, sd = [], []
            for n in sizes:
                v = [r["cost"] for r in rows if (r["kind"], r["cond"], r["n"], r["model"]) == (kind, cond, n, name)]
                if not v:
                    break
                mean.append(np.mean(v)); sd.append(np.std(v))
            if len(mean) == len(sizes):
                a.errorbar(sizes, mean, yerr=sd, color=col, ls=ls, marker="o", capsize=2,
                           lw=2.4 if name == "COBRA" else 1.4, label=name)
        a.set_xscale("log")
        a.set_xticks(sizes, [str(s) for s in sizes])
        a.set_xlabel("training sequences")
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("expected cost (lower is better)")
    ax[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Expected decision cost, mean +/- std over 5 seeds (synthetic data)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "cobra_benchmark.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true", help="only redraw the benchmark chart")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_benchmark()
    if args.no_train:
        return
    cobra, bayes = train_models()
    eps, labels = pick_episodes(cobra, np.random.default_rng(2024))
    fig_timeline(cobra, eps, labels)
    fig_density(cobra, eps, labels)
    fig_context(cobra, bayes)
    print("wrote", sorted(p.name for p in OUT.glob("cobra_*.png")))


if __name__ == "__main__":
    main()
