"""Quantum machine-learning classifiers: a variational (re-uploading) classifier and a quantum-kernel
classifier. Both follow the familiar fit / predict / predict_proba / score interface and run on the
built-in simulator; `to_qiskit` returns the circuit for one input for execution on Aer or hardware."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .ansatz import reuploading_classifier_circuit, zz_feature_map

__all__ = ['VariationalClassifier', 'QuantumKernel', 'QuantumKernelClassifier']


def _check_X(X):
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError('X must be a non-empty 2-D array (samples x features), got shape %s' % (X.shape,))
    if not np.isfinite(X).all():
        raise ValueError('X contains NaN or infinite values')
    return X


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
    C class labels and renormalised. Trained by minimising cross-entropy with exact gradients from the
    adjoint method (cost of about three simulations per step, samples processed in chunks) and L-BFGS.

    Parameters: layers (depth), n_qubits (default: number of features, at least ceil(log2 C)),
    reupload (re-encode the data in every layer), l2 (weight penalty), maxiter, seed.

    Size: simulation cost grows as 2^n_qubits. Up to about 10 qubits (features) and a few thousand
    samples train in seconds to minutes on a laptop; at most 22 qubits are accepted."""

    def __init__(self, layers=3, n_qubits=None, reupload=True, l2=1e-3, maxiter=200, seed=0):
        self.layers, self.n_qubits, self.reupload = layers, n_qubits, reupload
        self.l2, self.maxiter, self.seed = l2, maxiter, seed

    def _marginal_index(self, dim):
        return np.arange(dim) & (2 ** self.n_out - 1)

    def _class_probs(self, P):
        idx = self._marginal_index(P.shape[1]); marg = np.zeros((P.shape[0], 2 ** self.n_out))
        for k in range(2 ** self.n_out):
            marg[:, k] = P[:, idx == k].sum(1)
        return marg[:, :self.n_classes]

    def _probs(self, w, Xs):
        out = self._class_probs(self.circuit.probabilities(w, Xs))
        return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)

    def _objective(self, w, Xs, Y):
        """Mean cross-entropy and its exact gradient (adjoint method), without the l2 term."""
        B = len(Xs)

        def ce(psi, rows):
            # L_b = -sum_c Y_c log(q_c / S), q = class marginals, S = sum_c q_c;  phi = dL/d conj(psi)
            P = np.abs(psi) ** 2; q = np.clip(self._class_probs(P), 1e-12, None); S = q.sum(1, keepdims=True)
            Yr = Y[rows]
            L = -np.sum(Yr * np.log(q / S)) / B
            dq = np.zeros((len(rows), 2 ** self.n_out)); dq[:, :self.n_classes] = (-Yr / q + 1 / S) / B
            return L, dq[:, self._marginal_index(P.shape[1])] * psi
        return self.circuit.value_and_grad(w, Xs, ce)

    def fit(self, X, y):
        """Train on X (samples x features) and labels y; returns self."""
        X = _check_X(X); y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError('X has %d samples but y has %d' % (len(X), len(y)))
        self.classes_ = np.unique(y); self.n_classes = len(self.classes_)
        if self.n_classes < 2:
            raise ValueError('need at least two classes in y')
        self.n_out = max(1, int(np.ceil(np.log2(self.n_classes))))
        nq = self.n_qubits or max(X.shape[1], self.n_out)
        self.scaler = _Scaler().fit(X); Xs = self.scaler.transform(X)
        self.circuit = reuploading_classifier_circuit(X.shape[1], nq, self.layers, self.reupload)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        rng = np.random.default_rng(self.seed)
        w0 = rng.normal(0, 0.3, self.circuit.n_weights)
        self.history_ = []

        def loss_grad(w):
            L, g = self._objective(w, Xs, Y)
            L += self.l2 * w @ w; g = g + 2 * self.l2 * w
            self.history_.append(L)
            return L, g

        r = minimize(loss_grad, w0, jac=True, method='L-BFGS-B', options={'maxiter': self.maxiter})
        self.weights_ = r.x; self.loss_ = float(r.fun)
        return self

    def predict_proba(self, X):
        """Class probabilities (Born rule, restricted to the class labels)."""
        return self._probs(self.weights_, self.scaler.transform(X))

    def predict(self, X):
        """Most probable class for each sample."""
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def score(self, X, y):
        """Accuracy on (X, y)."""
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
        """Fit the feature scaling and build the feature map for X; returns self."""
        X = _check_X(X)
        self.scaler = _Scaler(0.0, self.scale * np.pi).fit(X)
        self.feature_map = zz_feature_map(X.shape[1], self.reps); return self

    def states(self, X):
        """Feature-map states of the rows of X, shape (samples, 2^features)."""
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
        """Solve the kernel ridge system for X and y; returns self."""
        X = _check_X(X); y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError('X has %d samples but y has %d' % (len(X), len(y)))
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError('need at least two classes in y')
        self.kernel.fit(X); self.X_ = X
        K = self.kernel(X)
        T = np.where(y[:, None] == self.classes_[None, :], 1.0, -1.0)
        self.alpha_ = np.linalg.solve(K + self.ridge * np.eye(len(X)), T)
        return self

    def decision_function(self, X):
        """Scores per class (one-versus-rest)."""
        return self.kernel(X, self.X_) @ self.alpha_

    def predict(self, X):
        """Class with the highest score for each sample."""
        return self.classes_[np.argmax(self.decision_function(X), 1)]

    def score(self, X, y):
        """Accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
