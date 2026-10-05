"""Quantum machine-learning classifiers: a variational (re-uploading) classifier and a quantum-kernel
classifier. Both follow the familiar fit / predict / predict_proba / score interface and run on the
built-in simulator; `to_qiskit` returns the circuit for one input for execution on Aer or hardware."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .ansatz import reuploading_classifier_circuit, zz_feature_map, shifted_weights

__all__ = ['VariationalClassifier', 'QuantumKernel', 'QuantumKernelClassifier']


class _Scaler:
    """Min-max scaling of each feature to [lo, hi] (angles)."""
    def __init__(self, lo=0.0, hi=np.pi):
        self.lo, self.hi = lo, hi

    def fit(self, X):
        self.mn, self.mx = X.min(0), X.max(0); return self

    def transform(self, X):
        span = np.where(self.mx > self.mn, self.mx - self.mn, 1.0)
        return self.lo + (self.hi - self.lo) * (np.asarray(X, float) - self.mn) / span


class VariationalClassifier:
    """Variational quantum classifier with data re-uploading.

    Each feature is encoded as a rotation angle; encoding and trainable layers alternate. Class
    probabilities are the Born-rule probabilities of the first ceil(log2 C) qubits, restricted to the
    C class labels and renormalised. Trained by minimising cross-entropy with exact parameter-shift
    gradients (L-BFGS).

    Parameters: layers (depth), n_qubits (default: number of features, at least ceil(log2 C)),
    reupload (re-encode the data in every layer), l2 (weight penalty), maxiter, seed."""

    def __init__(self, layers=3, n_qubits=None, reupload=True, l2=1e-3, maxiter=200, seed=0):
        self.layers, self.n_qubits, self.reupload = layers, n_qubits, reupload
        self.l2, self.maxiter, self.seed = l2, maxiter, seed

    def _probs(self, w, Xs):
        P = self.circuit.probabilities(w, Xs)                       # (B, 2^n); w: (K,) or (B, K)
        m = self.n_out; idx = np.arange(P.shape[1]) & (2 ** m - 1)
        marg = np.zeros((P.shape[0], 2 ** m))
        for k in range(2 ** m):
            marg[:, k] = P[:, idx == k].sum(1)
        out = marg[:, :self.n_classes]
        return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.classes_ = np.unique(y); self.n_classes = len(self.classes_)
        self.n_out = max(1, int(np.ceil(np.log2(self.n_classes))))
        nq = self.n_qubits or max(X.shape[1], self.n_out)
        self.scaler = _Scaler().fit(X); Xs = self.scaler.transform(X)
        self.circuit = reuploading_classifier_circuit(X.shape[1], nq, self.layers, self.reupload)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        rng = np.random.default_rng(self.seed)
        w0 = rng.normal(0, 0.3, self.circuit.n_weights)
        self.history_ = []

        def loss_grad(w):
            P = np.clip(self._probs(w, Xs), 1e-9, 1)
            L = -np.mean(np.sum(Y * np.log(P), 1)) + self.l2 * w @ w
            Ws = shifted_weights(w); B, K = len(Xs), len(w)        # all 2K shifts in one batch
            Pall = self._probs(np.repeat(Ws, B, axis=0), np.tile(Xs, (2 * K, 1))).reshape(2 * K, B, -1)
            dP = (Pall[0::2] - Pall[1::2]) / 2                      # (K, B, C)
            g = -np.mean(np.sum(Y[None] / P[None] * dP, 2), 1) + 2 * self.l2 * w
            self.history_.append(L)
            return L, g

        r = minimize(loss_grad, w0, jac=True, method='L-BFGS-B', options={'maxiter': self.maxiter})
        self.weights_ = r.x; self.loss_ = float(r.fun)
        return self

    def predict_proba(self, X):
        return self._probs(self.weights_, self.scaler.transform(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def to_qiskit(self, x, measure=True):
        """Circuit for one input sample, weights bound."""
        return self.circuit.to_qiskit(self.weights_, self.scaler.transform(np.atleast_2d(x))[0], measure)


class QuantumKernel:
    """Fidelity kernel k(x, x') = |<phi(x)|phi(x')>|^2 with the ZZ feature map (Havlicek et al. 2019).
    Use it with any kernel method, e.g. scikit-learn's SVC(kernel='precomputed')."""

    def __init__(self, reps=2, scale=1.0):
        self.reps, self.scale = reps, scale

    def fit(self, X):
        X = np.asarray(X, float)
        self.scaler = _Scaler(0.0, self.scale * np.pi).fit(X)
        self.feature_map = zz_feature_map(X.shape[1], self.reps); return self

    def states(self, X):
        return self.feature_map.state(None, self.scaler.transform(X))

    def __call__(self, X1, X2=None):
        S1 = self.states(X1); S2 = S1 if X2 is None else self.states(X2)
        return np.abs(S1.conj() @ S2.T) ** 2

    def to_qiskit(self, x1, x2):
        """Compute-uncompute circuit: P(all zeros) = k(x1, x2)."""
        a = self.feature_map.to_qiskit(None, self.scaler.transform(np.atleast_2d(x1))[0], measure=False)
        b = self.feature_map.to_qiskit(None, self.scaler.transform(np.atleast_2d(x2))[0], measure=False)
        qc = a.compose(b.inverse()); qc.measure_all(); return qc


class QuantumKernelClassifier:
    """Kernel ridge classifier (one-versus-rest, targets +-1) on the quantum kernel."""

    def __init__(self, reps=2, scale=1.0, ridge=1e-2):
        self.kernel = QuantumKernel(reps, scale); self.ridge = ridge

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.classes_ = np.unique(y); self.kernel.fit(X); self.X_ = X
        K = self.kernel(X)
        T = np.where(y[:, None] == self.classes_[None, :], 1.0, -1.0)
        self.alpha_ = np.linalg.solve(K + self.ridge * np.eye(len(X)), T)
        return self

    def decision_function(self, X):
        return self.kernel(X, self.X_) @ self.alpha_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), 1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
