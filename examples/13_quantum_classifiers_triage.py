"""Quantum, quantum-inspired and classical classifiers
===================================================

Medicine (simulated): quantum, quantum-inspired and classical classifiers on the same triage task.

A synthetic triage data set (four vital-sign features, two classes, simulated, not patient data) is
classified by a variational quantum classifier, a quantum-kernel classifier, a tensor-network (MPS)
classifier and two classical baselines: logistic regression on the raw features (linear) and on all
quadratic features (non-linear, the fair comparison here because the simulated risk is quadratic).
The comparison is repeated over 10 random data sets and splits (seeds) and reported as mean +-
standard deviation of test accuracy, because a single split is not enough to rank the methods. The variational circuit for one patient is then
executed on the Aer simulator to show that the exported circuit gives the same probability.
Use --quick for 2 seeds.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/q.png'
import numpy as np
from scipy.optimize import minimize
from quantum_mind.quantum import VariationalClassifier, QuantumKernelClassifier
from quantum_mind.inspired import MPSClassifier


# %%
# Simulated triage data and classical baselines
# ---------------------------------------------
# Four standardised features with a non-linear risk; logistic regression on raw and on quadratic
# features.
def make_data(seed, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))                                         # simulated, standardised vital signs
    risk = 1.4 * X[:, 0] - 1.1 * X[:, 1] ** 2 + 0.8 * X[:, 2] * X[:, 3]  # non-linear ground truth
    y = (risk + rng.normal(0, 0.5, n) > 0).astype(int)
    idx = rng.permutation(n)
    return X, y, idx[:220], idx[220:]


def quadratic(X):
    i, j = np.triu_indices(X.shape[1])
    return np.c_[X, X[:, i] * X[:, j]]


def logistic(Xtr, ytr, features=lambda X: X, l2=1e-2):
    A = np.c_[features(Xtr), np.ones(len(Xtr))]
    nll = lambda w: np.sum(np.logaddexp(0, A @ w) - ytr * (A @ w)) + l2 * w @ w      # noqa: E731
    w = minimize(nll, np.zeros(A.shape[1])).x
    return lambda Xq: (np.c_[features(Xq), np.ones(len(Xq))] @ w > 0).astype(int)


# %%
# Repeat over seeds
# -----------------
# Every method is trained and tested on the same splits; results are mean and standard deviation of
# test accuracy.
makers = {
    'variational quantum classifier (4 qubits)': lambda: VariationalClassifier(layers=3, maxiter=150),
    'quantum-kernel classifier (ZZ map)': lambda: QuantumKernelClassifier(reps=1, scale=0.5),
    'tensor-network MPS classifier': lambda: MPSClassifier(bond=6, local_dim=3, epochs=60),
}
BASELINE = 'logistic regression, linear (classical)'
QUAD = 'logistic regression, quadratic features (classical)'
seeds = range(2) if '--quick' in sys.argv else range(10)
scores = {name: [] for name in list(makers) + [BASELINE, QUAD]}
for seed in seeds:
    X, y, tr, te = make_data(seed)
    for name, make in makers.items():
        scores[name].append(make().fit(X[tr], y[tr]).score(X[te], y[te]))
    scores[BASELINE].append(float(np.mean(logistic(X[tr], y[tr])(X[te]) == y[te])))
    scores[QUAD].append(float(np.mean(logistic(X[tr], y[tr], quadratic)(X[te]) == y[te])))
print('test accuracy over %d seeds (mean +- sd):' % len(seeds))
for name, v in sorted(scores.items(), key=lambda kv: -np.mean(kv[1])):
    print('  %-52s %.3f +- %.3f' % (name, np.mean(v), np.std(v, ddof=1)))

# %%
# Run one circuit on Aer
# ----------------------
# The trained classifier's circuit for one patient, sampled on the Aer simulator, gives the same class
# probability as the built-in simulator.
X, y, tr, te = make_data(0)
vqc = makers['variational quantum classifier (4 qubits)']().fit(X[tr], y[tr])
try:
    from quantum_mind.circuits import run
    qc = vqc.to_qiskit(X[te[0]]); counts = run(qc, 'aer', 20000)
    p1 = sum(k for b, k in counts.items() if b[-1] == '1') / 20000
    print('one patient: simulator P(class 1) = %.3f, Aer circuit = %.3f' % (vqc.predict_proba(X[te[:1]])[0, 1], p1))
except ImportError:
    print('install qiskit-aer to run the circuit')
