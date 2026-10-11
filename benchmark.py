"""Benchmark COBRA against classical baselines on a synthetic robot-intent task.

Scenario: a robot watches a human for T frames and must pick one of
``handover`` / ``wait`` / ``ask``. Hidden intent is one of
``handover`` (0), ``retrieve`` (1: human reaches for own tool) or ``withdraw`` (2).
Frame features: gaze-like cue ``a``, hand cue ``b``, contact/force cue ``f``.

Generators
  context  - the meaning of ``b_t`` depends on the sign of the *previous* frame's
             ``a`` (``b_t ~ N(kappa_z * sign(a_{t-1}))``, kappa = +1 / -1 for
             handover / retrieve). Marginally ``b`` carries almost no class
             information, so a commutative filter or a linear model cannot use it.
  control  - features are conditionally independent given intent (a classical
             Bayes filter is near-optimal). Included so the comparison is not rigged.

An oracle that knows the generator gives the achievable bound. Every model is
fed through the same min-expected-cost decision rule with the same cost matrix.
Synthetic data only: results show behaviour on this task, not on real robots.

Usage: python benchmark.py [--quick] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from .baselines import BayesFilter, LogisticRegression, MLP
    from .cobra import COBRA, min_cost_action
except ImportError:  # run as a script
    from baselines import BayesFilter, LogisticRegression, MLP
    from cobra import COBRA, min_cost_action

T, M, D = 4, 3, 3
MU_A = np.array([0.7, -0.7, 0.0])
MU_F = np.array([0.0, 0.0, 0.9])
MU_B_CONTROL = np.array([0.6, -0.6, 0.0])
KAPPA = np.array([1.0, -1.0, 0.0])
SIGMA = 1.0
# cost[action, intent]; actions: handover, wait, ask
COST = np.array([[0.0, 6.0, 10.0],
                 [1.5, 0.0, 0.0],
                 [0.8, 0.8, 0.8]])
UNSAFE = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], bool)   # handing over to a non-receiver


def simulate(kind: str, n: int, rng: np.random.Generator, sigma: float = SIGMA):
    z = rng.integers(0, D, n)
    a = MU_A[z][:, None] + sigma * rng.standard_normal((n, T))
    f = MU_F[z][:, None] + sigma * rng.standard_normal((n, T))
    if kind == "context":
        prev = np.concatenate([np.zeros((n, 1)), a[:, :-1]], axis=1)
        b = KAPPA[z][:, None] * np.sign(prev) + sigma * rng.standard_normal((n, T))
    elif kind == "control":
        b = MU_B_CONTROL[z][:, None] + sigma * rng.standard_normal((n, T))
    else:
        raise ValueError(kind)
    return np.stack([a, b, f], axis=2), z


def oracle_proba(kind: str, x: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    a, b, f = x[..., 0], x[..., 1], x[..., 2]
    ll = np.zeros((len(x), D))
    for k in range(D):
        l = -0.5 * ((a - MU_A[k]) ** 2 + (f - MU_F[k]) ** 2) / sigma ** 2
        if kind == "context":
            prev = np.concatenate([np.zeros((len(x), 1)), a[:, :-1]], axis=1)
            l += -0.5 * (b - KAPPA[k] * np.sign(prev)) ** 2 / sigma ** 2
        else:
            l += -0.5 * (b - MU_B_CONTROL[k]) ** 2 / sigma ** 2
        ll[:, k] = l.sum(1)
    ll -= ll.max(1, keepdims=True)
    p = np.exp(ll)
    return p / p.sum(1, keepdims=True)


def perturb(x: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Test-time stress: 'dropout' zero-fills 25% of frames' a/b cues."""
    if kind != "dropout":
        return x
    x = x.copy()
    mask = rng.random(x.shape[:2]) < 0.25
    x[mask, 0] = 0.0
    x[mask, 1] = 0.0
    return x


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of an error > score of a correct case); 0.5 = uninformative."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(p: np.ndarray, z: np.ndarray) -> dict[str, float]:
    n = len(z)
    p = np.clip(p, 1e-9, 1)
    p = p / p.sum(1, keepdims=True)
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == z
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if np.any(bins == b))
    act = min_cost_action(p, COST)
    ent = -(p * np.log(p)).sum(1)
    return {
        "acc": float(correct.mean()),
        "nll": float(-np.log(p[np.arange(n), z]).mean()),
        "ece": float(ece),
        "cost": float(COST[act, z].mean()),
        "unsafe": float(UNSAFE[act, z].mean()),
        "ask": float((act == 2).mean()),
        "err_auroc": auroc(ent, ~correct),
    }


def make_models(seed: int, steps: int):
    return {
        "COBRA": COBRA(D, M, steps=steps, seed=seed, restarts=8, batch=100),
        "COBRA (no coherence)": None,   # filled below: dephasing pinned to 1
        "Bayes filter": BayesFilter(D),
        "Logistic reg.": LogisticRegression(D, epochs=400, seed=seed),
        "MLP (32 hidden)": MLP(D, hidden=32, epochs=500, seed=seed),
    }


class _NoCoherence(COBRA):
    """Ablation: dephasing pinned to ~1, so rotations are erased every frame."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.theta[-1] = 12.0

    def _loss(self, theta, x, y):
        theta = theta.copy()
        theta[-1] = 12.0
        return super()._loss(theta, x, y)

    def _forward(self, theta, x, *, trace=False):
        theta = theta.copy()
        theta[-1] = 12.0
        return super()._forward(theta, x, trace=trace)


def run_cell(job):
    kind, n_train, seed, steps = job
    rng = np.random.default_rng(1000 * seed + n_train)
    xtr, ztr = simulate(kind, n_train, rng)
    models = make_models(seed, steps)
    models["COBRA (no coherence)"] = _NoCoherence(D, M, steps=steps, seed=seed, restarts=1, batch=100)
    rows = []
    t_rng = np.random.default_rng(777 + seed)
    xte, zte = simulate(kind, 3000, t_rng)
    xte_hi, zte_hi = simulate(kind, 3000, t_rng, sigma=1.5 * SIGMA)
    tests = {
        "clean": (xte, zte, oracle_proba(kind, xte)),
        "noise x1.5": (xte_hi, zte_hi, oracle_proba(kind, xte_hi, 1.5 * SIGMA)),
        "dropout": None,
    }
    xd = perturb(xte, "dropout", np.random.default_rng(5 + seed))
    tests["dropout"] = (xd, zte, None)
    for name, model in models.items():
        t0 = time.time()
        model.fit(xtr, ztr)
        fit_s = time.time() - t0
        for cond, (xx, zz, _) in tests.items():
            m = metrics(model.predict_proba(xx), zz)
            if name.startswith("COBRA") and cond == "clean":
                coh = model.coherence(xx)
                correct = model.predict_proba(xx).argmax(1) == zz
                m["coh_auroc"] = auroc(coh, ~correct)
            rows.append(dict(kind=kind, n=n_train, seed=seed, model=name, cond=cond,
                             params=getattr(model, "n_params", 0), fit_s=fit_s, **m))
    for cond, (xx, zz, po) in tests.items():
        if po is not None:
            rows.append(dict(kind=kind, n=n_train, seed=seed, model="Oracle", cond=cond,
                             params=0, fit_s=0.0, **metrics(po, zz)))
    return rows


def aggregate(rows):
    keys = sorted({(r["kind"], r["cond"], r["n"], r["model"]) for r in rows})
    out = {}
    for k in keys:
        sel = [r for r in rows if (r["kind"], r["cond"], r["n"], r["model"]) == k]
        agg = {}
        for f in ("acc", "nll", "ece", "cost", "unsafe", "ask", "err_auroc", "coh_auroc", "params", "fit_s"):
            v = [r[f] for r in sel if f in r and not np.isnan(r[f])]
            if v:
                agg[f] = (float(np.mean(v)), float(np.std(v)))
        out[k] = agg
    return out


def to_markdown(agg, sizes, seeds):
    order = ["Oracle", "COBRA", "COBRA (no coherence)", "Bayes filter", "Logistic reg.", "MLP (32 hidden)"]
    lines = [f"# COBRA benchmark results\n",
             f"Mean ± std over {seeds} seeds; 3000 test sequences per cell; T={T} frames. "
             "Cost uses the matrix in `benchmark.py` (lower is better). "
             "Oracle knows the true generator. Synthetic data only.\n"]
    for kind in ("context", "control"):
        for cond in ("clean", "noise x1.5", "dropout"):
            lines.append(f"\n## Generator: {kind} · test condition: {cond}\n")
            for n in sizes:
                lines.append(f"\n**Training sequences: {n}**\n")
                lines.append("| Model | Params | Accuracy | NLL | ECE | Expected cost | Unsafe handover rate | Ask rate |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for name in order:
                    a = agg.get((kind, cond, n, name))
                    if not a:
                        continue
                    f = lambda key, d=3: f"{a[key][0]:.{d}f} ± {a[key][1]:.{d}f}" if key in a else "–"
                    lines.append(f"| {name} | {int(a['params'][0]) if 'params' in a else 0} | {f('acc')} | {f('nll')} | "
                                 f"{f('ece')} | {f('cost')} | {f('unsafe')} | {f('ask')} |")
    lines.append("\n## Error-detection AUROC (clean test, largest training size)\n")
    lines.append("Higher means the signal better flags the model's own mistakes (0.5 = useless).\n")
    lines.append("| Generator | Model | AUROC of belief entropy | AUROC of l1 coherence |")
    lines.append("|---|---|---|---|")
    n = sizes[-1]
    for kind in ("context", "control"):
        for name in ("COBRA", "COBRA (no coherence)", "Bayes filter", "MLP (32 hidden)", "Oracle"):
            a = agg.get((kind, "clean", n, name))
            if a:
                e = f"{a['err_auroc'][0]:.3f}" if "err_auroc" in a else "–"
                c = f"{a['coh_auroc'][0]:.3f}" if "coh_auroc" in a else "–"
                lines.append(f"| {kind} | {name} | {e} | {c} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, small sizes (smoke test)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sizes = [60, 250, 1000]
    seeds = args.seeds
    steps = args.steps
    if args.quick:
        sizes, seeds, steps = [250], 1, 40
    jobs = [(k, n, s, steps) for k in ("context", "control") for n in sizes for s in range(seeds)]
    t0 = time.time()
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for part in pool.imap_unordered(run_cell, jobs) for r in part]
    agg = aggregate(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.md").write_text(to_markdown(agg, sizes, seeds))
    (out / "benchmark_raw.json").write_text(json.dumps(rows, indent=1))
    print(to_markdown(agg, sizes, seeds))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
