"""Medicine (simulated): quantum, quantum-inspired and classical classifiers on the same triage task.

A synthetic triage data set (four vital-sign features, two classes, simulated, not patient data) is
classified by a variational quantum classifier, a quantum-kernel classifier, a tensor-network (MPS)
classifier and a classical logistic-regression baseline. The variational circuit for one patient is
then executed on the Aer simulator to show that the exported circuit gives the same probability."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from scipy.optimize import minimize
from qlcog.quantum import VariationalClassifier, QuantumKernelClassifier
from qlcog.inspired import MPSClassifier

rng = np.random.default_rng(1)
n = 300
X = rng.normal(0, 1, (n, 4))                                         # simulated, standardised vital signs
risk = 1.4 * X[:, 0] - 1.1 * X[:, 1] ** 2 + 0.8 * X[:, 2] * X[:, 3]  # non-linear ground truth
y = (risk + rng.normal(0, 0.5, n) > 0).astype(int)
idx = rng.permutation(n); tr, te = idx[:220], idx[220:]


def logistic(Xtr, ytr):
    A = np.c_[Xtr, np.ones(len(Xtr))]
    nll = lambda w: np.sum(np.logaddexp(0, A @ w) - ytr * (A @ w))      # noqa: E731
    w = minimize(nll, np.zeros(A.shape[1])).x
    return lambda Xq: (np.c_[Xq, np.ones(len(Xq))] @ w > 0).astype(int)


models = {
    'variational quantum classifier (4 qubits)': VariationalClassifier(layers=3, maxiter=150),
    'quantum-kernel classifier (ZZ map)': QuantumKernelClassifier(reps=1, scale=0.5),
    'tensor-network MPS classifier': MPSClassifier(bond=6, local_dim=3, epochs=60),
}
for name, m in models.items():
    m.fit(X[tr], y[tr]); print('%-44s test accuracy %.3f' % (name, m.score(X[te], y[te])))
pred = logistic(X[tr], y[tr]); print('%-44s test accuracy %.3f' % ('logistic regression (classical baseline)', np.mean(pred(X[te]) == y[te])))

vqc = models['variational quantum classifier (4 qubits)']
try:
    from qlcog.circuits import run
    qc = vqc.to_qiskit(X[te[0]]); counts = run(qc, 'aer', 20000)
    p1 = sum(k for b, k in counts.items() if b[-1] == '1') / 20000
    print('one patient: simulator P(class 1) = %.3f, Aer circuit = %.3f' % (vqc.predict_proba(X[te[:1]])[0, 1], p1))
except ImportError:
    print('install qiskit-aer to run the circuit')
