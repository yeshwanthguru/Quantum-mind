"""Quantum machine-learning classifiers: a variational (re-uploading) classifier and a quantum-kernel
classifier. Both follow the familiar fit / predict / predict_proba / score interface and run on the
built-in simulator; `to_qiskit` returns the circuit for one input for execution on Aer or hardware."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .ansatz import reuploading_classifier_circuit, zz_feature_map

__all__ = ['VariationalClassifier', 'VariationalRegressor', 'QuantumKernel', 'QuantumKernelClassifier',
           'QuantumKernelAnomalyDetector', 'QuantumKernelClustering', 'rbf_kernel']


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


class VariationalRegressor:
    """Variational (data re-uploading) regressor: prediction = scale * <Z_0> + offset, with scale and
    offset set from the training targets; mean squared error minimised with exact adjoint gradients.
    Used for small forecasting tasks (inputs = a window of past values). Same size limits as
    VariationalClassifier."""

    def __init__(self, layers=3, n_qubits=None, reupload=True, l2=1e-3, maxiter=200, seed=0):
        self.layers, self.n_qubits, self.reupload = layers, n_qubits, reupload
        self.l2, self.maxiter, self.seed = l2, maxiter, seed

    def _z(self, P):
        sign = 1 - 2 * (np.arange(P.shape[1]) & 1)            # <Z_0>
        return P @ sign, sign

    def fit(self, X, y):
        """Train on X (samples x features) and real targets y; returns self."""
        X = _check_X(X); y = np.asarray(y, float)
        if len(y) != len(X):
            raise ValueError('X has %d samples but y has %d' % (len(X), len(y)))
        self.offset = float((y.max() + y.min()) / 2); self.scale = float(max((y.max() - y.min()) / 2, 1e-12)) * 1.1
        t = (y - self.offset) / self.scale                    # targets in (-1, 1)
        nq = self.n_qubits or X.shape[1]
        self.scaler = _Scaler().fit(X); Xs = self.scaler.transform(X)
        self.circuit = reuploading_classifier_circuit(X.shape[1], nq, self.layers, self.reupload)
        B = len(Xs); rng = np.random.default_rng(self.seed); self.history_ = []

        def mse(psi, rows):
            P = np.abs(psi) ** 2; z, sign = self._z(P); r = z - t[rows]
            return float(np.sum(r ** 2) / B), (2 * r / B)[:, None] * sign[None, :] * psi

        def loss_grad(w):
            L, g = self.circuit.value_and_grad(w, Xs, mse)
            L += self.l2 * w @ w; g = g + 2 * self.l2 * w; self.history_.append(L)
            return L, g
        r = minimize(loss_grad, rng.normal(0, 0.3, self.circuit.n_weights), jac=True, method='L-BFGS-B',
                     options={'maxiter': self.maxiter})
        self.weights_ = r.x; self.loss_ = float(r.fun)
        return self

    def predict(self, X):
        """Predicted values."""
        z, _ = self._z(self.circuit.probabilities(self.weights_, self.scaler.transform(X)))
        return self.offset + self.scale * z

    def score(self, X, y):
        """Coefficient of determination R^2."""
        y = np.asarray(y, float); e = self.predict(X) - y
        return float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))

    def to_qiskit(self, x, measure=True):
        """Circuit for one input (weights bound); <Z_0> of qubit 0 gives the prediction."""
        return self.circuit.to_qiskit(self.weights_, self.scaler.transform(np.atleast_2d(x))[0], measure)


def rbf_kernel(X1, X2=None, gamma=1.0):
    """Classical Gaussian (RBF) kernel exp(-gamma ||x - x'||^2), the baseline for the quantum kernel."""
    X1 = np.asarray(X1, float); X2 = X1 if X2 is None else np.asarray(X2, float)
    d = (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)[None, :] - 2 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(d, 0))


class _KernelMixin:
    def _kernel_fit(self, X):
        X = _check_X(X)
        if self.kernel == 'quantum':
            self.qk = QuantumKernel(self.reps, self.scale).fit(X)
            self.K = lambda A, B=None: self.qk(A, B)          # noqa: E731
        elif self.kernel == 'rbf':
            mu, sd = X.mean(0), X.std(0) + 1e-12
            self.K = lambda A, B=None: rbf_kernel((A - mu) / sd, None if B is None else (B - mu) / sd, self.gamma)   # noqa: E731
        else:
            raise ValueError("kernel must be 'quantum' or 'rbf'")
        return X


class QuantumKernelAnomalyDetector(_KernelMixin):
    """Anomaly score = squared distance, in the kernel's feature space, from a point's feature vector
    to the mean feature vector of the (normal) training data: k(x, x) - 2 mean_i k(x, x_i) + const.
    The threshold is the `quantile` of the training scores. kernel='quantum' (ZZ feature map) or 'rbf'
    (classical baseline with the same rule)."""

    def __init__(self, kernel='quantum', reps=1, scale=0.5, gamma=0.5, quantile=0.95):
        self.kernel, self.reps, self.scale, self.gamma, self.quantile = kernel, reps, scale, gamma, quantile

    def fit(self, X):
        """Learn the normal data; returns self."""
        self.X_ = self._kernel_fit(X)
        self.mean_k_ = float(self.K(self.X_).mean())
        self.threshold_ = float(np.quantile(self.score_samples(self.X_), self.quantile))
        return self

    def score_samples(self, X):
        """Anomaly score of each sample (higher = more anomalous)."""
        X = _check_X(X); diag = np.array([self.K(x[None])[0, 0] for x in X])
        return diag - 2 * self.K(X, self.X_).mean(1) + self.mean_k_

    def predict(self, X):
        """1 for anomalies (score above the threshold), 0 otherwise."""
        return (self.score_samples(X) > self.threshold_).astype(int)


class QuantumKernelClustering(_KernelMixin):
    """Spectral clustering on a kernel: normalised affinity, its leading eigenvectors, then k-means
    (several restarts). kernel='quantum' or 'rbf' (classical baseline)."""

    def __init__(self, n_clusters=2, kernel='quantum', reps=1, scale=0.5, gamma=0.5, restarts=10, seed=0):
        self.k, self.kernel, self.reps, self.scale, self.gamma = n_clusters, kernel, reps, scale, gamma
        self.restarts, self.seed = restarts, seed

    def fit_predict(self, X):
        """Cluster labels for X."""
        X = self._kernel_fit(X); Kx = self.K(X); np.fill_diagonal(Kx, 0)
        d = Kx.sum(1); Dm = 1 / np.sqrt(np.maximum(d, 1e-12))
        w, V = np.linalg.eigh(Dm[:, None] * Kx * Dm[None, :])
        E = V[:, -self.k:]; E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        rng = np.random.default_rng(self.seed); best = (np.inf, None)
        for _ in range(self.restarts):
            C = E[rng.choice(len(E), self.k, replace=False)]
            for _ in range(100):
                lab = np.argmin(((E[:, None, :] - C[None]) ** 2).sum(-1), 1)
                newC = np.array([E[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(self.k)])
                if np.allclose(newC, C):
                    break
                C = newC
            inertia = float(((E - C[lab]) ** 2).sum())
            if inertia < best[0]:
                best = (inertia, lab)
        self.labels_ = best[1]
        return self.labels_
