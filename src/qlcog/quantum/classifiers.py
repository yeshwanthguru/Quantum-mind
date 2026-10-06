"""Quantum machine-learning models on the built-in simulator.

* :class:`VariationalClassifier` and :class:`VariationalRegressor`: data re-uploading circuits
  trained with exact adjoint gradients and L-BFGS.
* :class:`QuantumKernel`: fidelity kernel with the ZZ feature map, usable by any kernel method.
* :class:`QuantumKernelClassifier`: kernel ridge classification on that kernel.
* :class:`QuantumKernelAnomalyDetector` and :class:`QuantumKernelClustering`: one-class anomaly
  scores and spectral clustering, each with the classical RBF kernel (:func:`rbf_kernel`) as baseline.

All follow the familiar scikit-learn interface (``fit``, ``predict``, ``score``); ``to_qiskit``
returns the circuit for one input for execution on Aer or hardware.

Examples
--------
>>> import numpy as np
>>> from qlcog.quantum import VariationalClassifier
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(80, 2)); y = (X[:, 0] * X[:, 1] > 0).astype(int)
>>> clf = VariationalClassifier(layers=2, maxiter=60).fit(X, y)
>>> clf.predict(X[:5]).shape
(5,)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .ansatz import reuploading_classifier_circuit, zz_feature_map

__all__ = ['VariationalClassifier', 'VariationalRegressor', 'QuantumKernel', 'QuantumKernelClassifier',
           'QuantumKernelAnomalyDetector', 'QuantumKernelClustering', 'rbf_kernel']


def _check_X(X):
    """Validate a non-empty, finite 2-D sample matrix."""
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError('X must be a non-empty 2-D array (samples x features), got shape %s' % (X.shape,))
    if not np.isfinite(X).all():
        raise ValueError('X contains NaN or infinite values')
    return X


class _Scaler:
    """Min-max scaling of each feature to [lo, hi] (rotation angles)."""
    def __init__(self, lo=0.0, hi=np.pi):
        self.lo, self.hi = lo, hi

    def fit(self, X):
        """Record the feature ranges of X."""
        self.mn, self.mx = X.min(0), X.max(0)
        return self

    def transform(self, X):
        """Map X into [lo, hi] (constant features map to lo)."""
        span = np.where(self.mx > self.mn, self.mx - self.mn, 1.0)
        return self.lo + (self.hi - self.lo) * (np.asarray(X, float) - self.mn) / span


class VariationalClassifier:
    """Variational quantum classifier with data re-uploading.

    Each feature is encoded as a rotation angle, and encoding and trainable layers alternate. Class
    probabilities are the Born-rule probabilities of the first :math:`\\lceil\\log_2 C\\rceil` qubits,
    restricted to the C class labels and renormalised. Training minimises cross-entropy with exact
    gradients from the adjoint method (about three simulations per step, samples processed in chunks)
    and L-BFGS.

    Simulation cost grows as :math:`2^{n\\_qubits}`. Up to about 10 qubits (features) and a few
    thousand samples train in seconds to minutes on a laptop; at most 22 qubits are accepted.

    Parameters
    ----------
    layers : int, optional
        Number of encoding-plus-trainable layers.
    n_qubits : int, optional
        Number of qubits (default: number of features, at least :math:`\\lceil\\log_2 C\\rceil`).
    reupload : bool, optional
        Re-encode the data in every layer.
    l2 : float, optional
        Weight penalty.
    maxiter : int, optional
        L-BFGS iterations.
    seed : int, optional
        Seed of the initial weights.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class labels.
    weights_ : numpy.ndarray
        Trained weights.
    loss_ : float
        Final training loss.
    history_ : list of float
        Loss at every evaluation.
    circuit : Circuit
        The trained circuit.
    """

    def __init__(self, layers=3, n_qubits=None, reupload=True, l2=1e-3, maxiter=200, seed=0):
        self.layers, self.n_qubits, self.reupload = layers, n_qubits, reupload
        self.l2, self.maxiter, self.seed = l2, maxiter, seed

    def _marginal_index(self, dim):
        """Value of the read-out qubits for each basis state."""
        return np.arange(dim) & (2 ** self.n_out - 1)

    def _class_probs(self, P):
        """Marginal probabilities of the read-out qubits, restricted to the class labels."""
        idx = self._marginal_index(P.shape[1])
        marg = np.zeros((P.shape[0], 2 ** self.n_out))
        for k in range(2 ** self.n_out):
            marg[:, k] = P[:, idx == k].sum(1)
        return marg[:, :self.n_classes]

    def _probs(self, w, Xs):
        """Renormalised class probabilities for scaled inputs."""
        out = self._class_probs(self.circuit.probabilities(w, Xs))
        return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)

    def _objective(self, w, Xs, Y):
        """Mean cross-entropy and its exact gradient (adjoint method), without the l2 term."""
        B = len(Xs)

        def ce(psi, rows):
            # L_b = -sum_c Y_c log(q_c / S), q = class marginals, S = sum_c q_c;  phi = dL/d conj(psi)
            P = np.abs(psi) ** 2
            q = np.clip(self._class_probs(P), 1e-12, None)
            S = q.sum(1, keepdims=True)
            Yr = Y[rows]
            L = -np.sum(Yr * np.log(q / S)) / B
            dq = np.zeros((len(rows), 2 ** self.n_out))
            dq[:, :self.n_classes] = (-Yr / q + 1 / S) / B
            return L, dq[:, self._marginal_index(P.shape[1])] * psi
        return self.circuit.value_and_grad(w, Xs, ce)

    def fit(self, X, y):
        """Train the classifier.

        Parameters
        ----------
        X : array_like
            Samples, shape ``(n_samples, n_features)``.
        y : array_like
            Labels, at least two classes.

        Returns
        -------
        VariationalClassifier
            ``self``.

        Raises
        ------
        ValueError
            On malformed input or fewer than two classes.
        """
        X = _check_X(X)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError('X has %d samples but y has %d' % (len(X), len(y)))
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        if self.n_classes < 2:
            raise ValueError('need at least two classes in y')
        self.n_out = max(1, int(np.ceil(np.log2(self.n_classes))))      # read-out qubits
        nq = self.n_qubits or max(X.shape[1], self.n_out)
        self.scaler = _Scaler().fit(X)
        Xs = self.scaler.transform(X)
        self.circuit = reuploading_classifier_circuit(X.shape[1], nq, self.layers, self.reupload)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)        # one-hot targets
        rng = np.random.default_rng(self.seed)
        w0 = rng.normal(0, 0.3, self.circuit.n_weights)
        self.history_ = []

        def loss_grad(w):
            L, g = self._objective(w, Xs, Y)
            L += self.l2 * w @ w
            g = g + 2 * self.l2 * w
            self.history_.append(L)
            return L, g

        r = minimize(loss_grad, w0, jac=True, method='L-BFGS-B', options={'maxiter': self.maxiter})
        self.weights_ = r.x
        self.loss_ = float(r.fun)
        return self

    def predict_proba(self, X):
        """Class probabilities.

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_classes)``; Born-rule probabilities restricted to the class labels.
        """
        return self._probs(self.weights_, self.scaler.transform(X))

    def predict(self, X):
        """Most probable class for each sample.

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
        """
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def score(self, X, y):
        """Accuracy.

        Parameters
        ----------
        X : array_like
            Samples.
        y : array_like
            True labels.

        Returns
        -------
        float
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def to_qiskit(self, x, measure=True):
        """Circuit for one input sample, with the trained weights bound.

        Parameters
        ----------
        x : array_like
            One sample (unscaled).
        measure : bool, optional
            Add measurements.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.circuit.to_qiskit(self.weights_, self.scaler.transform(np.atleast_2d(x))[0], measure)


class QuantumKernel:
    """Fidelity quantum kernel with the ZZ feature map (Havlíček et al., 2019).

    .. math:: k(x, x') = |\\langle\\phi(x)|\\phi(x')\\rangle|^2

    Use it with any kernel method, for example scikit-learn's ``SVC(kernel='precomputed')``.

    Parameters
    ----------
    reps : int, optional
        Repetitions of the feature map.
    scale : float, optional
        Features are scaled to ``[0, scale * pi]``.
    """

    def __init__(self, reps=2, scale=1.0):
        self.reps, self.scale = reps, scale

    def fit(self, X):
        """Fit the feature scaling and build the feature map.

        Parameters
        ----------
        X : array_like
            Training samples.

        Returns
        -------
        QuantumKernel
            ``self``.
        """
        X = _check_X(X)
        self.scaler = _Scaler(0.0, self.scale * np.pi).fit(X)
        self.feature_map = zz_feature_map(X.shape[1], self.reps)
        return self

    def states(self, X):
        """Feature-map states.

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, 2**n_features)``.
        """
        return self.feature_map.state(None, self.scaler.transform(X))

    def __call__(self, X1, X2=None):
        """Gram matrix ``k(X1, X2)`` (``X2`` defaults to ``X1``)."""
        S1 = self.states(X1)
        S2 = S1 if X2 is None else self.states(X2)
        return np.abs(S1.conj() @ S2.T) ** 2

    def to_qiskit(self, x1, x2):
        """Compute-uncompute circuit whose all-zeros probability is ``k(x1, x2)``.

        Parameters
        ----------
        x1, x2 : array_like
            Two samples.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        a = self.feature_map.to_qiskit(None, self.scaler.transform(np.atleast_2d(x1))[0], measure=False)
        b = self.feature_map.to_qiskit(None, self.scaler.transform(np.atleast_2d(x2))[0], measure=False)
        qc = a.compose(b.inverse())
        qc.measure_all()
        return qc


class QuantumKernelClassifier:
    """Kernel ridge classifier on the quantum kernel (one-versus-rest, targets +-1).

    Parameters
    ----------
    reps : int, optional
        Repetitions of the feature map.
    scale : float, optional
        Feature scale (see :class:`QuantumKernel`).
    ridge : float, optional
        Ridge regularisation.
    """

    def __init__(self, reps=2, scale=1.0, ridge=1e-2):
        self.kernel = QuantumKernel(reps, scale)
        self.ridge = ridge

    def fit(self, X, y):
        """Solve the kernel ridge system.

        Parameters
        ----------
        X : array_like
            Samples.
        y : array_like
            Labels.

        Returns
        -------
        QuantumKernelClassifier
            ``self``.
        """
        X = _check_X(X)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError('X has %d samples but y has %d' % (len(X), len(y)))
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError('need at least two classes in y')
        self.kernel.fit(X)
        self.X_ = X
        K = self.kernel(X)
        T = np.where(y[:, None] == self.classes_[None, :], 1.0, -1.0)
        self.alpha_ = np.linalg.solve(K + self.ridge * np.eye(len(X)), T)    # (K + r I) alpha = T
        return self

    def decision_function(self, X):
        """Scores per class (one-versus-rest).

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_classes)``.
        """
        return self.kernel(X, self.X_) @ self.alpha_

    def predict(self, X):
        """Class with the highest score for each sample."""
        return self.classes_[np.argmax(self.decision_function(X), 1)]

    def score(self, X, y):
        """Accuracy on ``(X, y)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


class VariationalRegressor:
    """Variational (data re-uploading) regressor.

    The prediction is ``scale * <Z_0> + offset``, with ``scale`` and ``offset`` set from the range of
    the training targets. Mean squared error is minimised with exact adjoint gradients. Suited to small
    forecasting tasks (inputs = a window of past values). Same size limits as
    :class:`VariationalClassifier`.

    Parameters
    ----------
    layers, n_qubits, reupload, l2, maxiter, seed
        As in :class:`VariationalClassifier`.
    """

    def __init__(self, layers=3, n_qubits=None, reupload=True, l2=1e-3, maxiter=200, seed=0):
        self.layers, self.n_qubits, self.reupload = layers, n_qubits, reupload
        self.l2, self.maxiter, self.seed = l2, maxiter, seed

    def _z(self, P):
        """<Z_0> from probabilities, and the sign vector used for the gradient."""
        sign = 1 - 2 * (np.arange(P.shape[1]) & 1)            # <Z_0>
        return P @ sign, sign

    def fit(self, X, y):
        """Train the regressor.

        Parameters
        ----------
        X : array_like
            Samples, shape ``(n_samples, n_features)``.
        y : array_like
            Real targets.

        Returns
        -------
        VariationalRegressor
            ``self``.
        """
        X = _check_X(X)
        y = np.asarray(y, float)
        if len(y) != len(X):
            raise ValueError('X has %d samples but y has %d' % (len(X), len(y)))
        self.offset = float((y.max() + y.min()) / 2)
        self.scale = float(max((y.max() - y.min()) / 2, 1e-12)) * 1.1
        t = (y - self.offset) / self.scale                    # targets in (-1, 1)
        nq = self.n_qubits or X.shape[1]
        self.scaler = _Scaler().fit(X)
        Xs = self.scaler.transform(X)
        self.circuit = reuploading_classifier_circuit(X.shape[1], nq, self.layers, self.reupload)
        B = len(Xs)
        rng = np.random.default_rng(self.seed)
        self.history_ = []

        def mse(psi, rows):
            P = np.abs(psi) ** 2
            z, sign = self._z(P)
            r = z - t[rows]
            return float(np.sum(r ** 2) / B), (2 * r / B)[:, None] * sign[None, :] * psi

        def loss_grad(w):
            L, g = self.circuit.value_and_grad(w, Xs, mse)
            L += self.l2 * w @ w
            g = g + 2 * self.l2 * w
            self.history_.append(L)
            return L, g
        r = minimize(loss_grad, rng.normal(0, 0.3, self.circuit.n_weights), jac=True, method='L-BFGS-B',
                     options={'maxiter': self.maxiter})
        self.weights_ = r.x
        self.loss_ = float(r.fun)
        return self

    def predict(self, X):
        """Predicted values.

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
        """
        z, _ = self._z(self.circuit.probabilities(self.weights_, self.scaler.transform(X)))
        return self.offset + self.scale * z

    def score(self, X, y):
        """Coefficient of determination :math:`R^2`.

        Parameters
        ----------
        X : array_like
            Samples.
        y : array_like
            True targets.

        Returns
        -------
        float
        """
        y = np.asarray(y, float)
        e = self.predict(X) - y
        return float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))

    def to_qiskit(self, x, measure=True):
        """Circuit for one input with the trained weights bound; ``<Z_0>`` gives the prediction.

        Parameters
        ----------
        x : array_like
            One sample (unscaled).
        measure : bool, optional
            Add measurements.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.circuit.to_qiskit(self.weights_, self.scaler.transform(np.atleast_2d(x))[0], measure)


def rbf_kernel(X1, X2=None, gamma=1.0):
    """Classical Gaussian (RBF) kernel, the baseline for the quantum kernel.

    Parameters
    ----------
    X1 : array_like
        Samples, shape ``(n1, d)``.
    X2 : array_like, optional
        Samples, shape ``(n2, d)`` (default ``X1``).
    gamma : float, optional
        Inverse squared length scale.

    Returns
    -------
    numpy.ndarray
        :math:`\\exp(-\\gamma \\lVert x - x' \\rVert^2)`, shape ``(n1, n2)``.
    """
    X1 = np.asarray(X1, float)
    X2 = X1 if X2 is None else np.asarray(X2, float)
    d = (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)[None, :] - 2 * X1 @ X2.T     # squared distances
    return np.exp(-gamma * np.maximum(d, 0))


class _KernelMixin:
    """Shared kernel set-up: 'quantum' (ZZ feature map) or 'rbf' on standardised features."""

    def _kernel_fit(self, X):
        """Validate X and build self.K, a function (A, B=None) -> Gram matrix."""
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
    """Kernel anomaly detector (distance to the mean in feature space).

    The anomaly score is the squared feature-space distance from a point to the mean feature vector
    of the normal training data, :math:`k(x, x) - 2\\,\\overline{k(x, x_i)} + \\text{const}`. The
    threshold is the ``quantile`` of the training scores.

    Angle encoding is periodic, so points far outside the training range can wrap around and look
    normal; the RBF baseline does not have this problem (see example 29).

    Parameters
    ----------
    kernel : {'quantum', 'rbf'}, optional
        Kernel; ``'rbf'`` is the classical baseline with the same rule.
    reps : int, optional
        Feature-map repetitions (quantum kernel).
    scale : float, optional
        Feature scale (quantum kernel).
    gamma : float, optional
        RBF parameter.
    quantile : float, optional
        Quantile of the training scores used as threshold.
    """

    def __init__(self, kernel='quantum', reps=1, scale=0.5, gamma=0.5, quantile=0.95):
        self.kernel, self.reps, self.scale, self.gamma, self.quantile = kernel, reps, scale, gamma, quantile

    def fit(self, X):
        """Learn the normal data.

        Parameters
        ----------
        X : array_like
            Normal samples.

        Returns
        -------
        QuantumKernelAnomalyDetector
            ``self``.
        """
        self.X_ = self._kernel_fit(X)
        self.mean_k_ = float(self.K(self.X_).mean())
        self.threshold_ = float(np.quantile(self.score_samples(self.X_), self.quantile))
        return self

    def score_samples(self, X):
        """Anomaly score of each sample (higher is more anomalous).

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
        """
        X = _check_X(X)
        diag = np.array([self.K(x[None])[0, 0] for x in X])
        return diag - 2 * self.K(X, self.X_).mean(1) + self.mean_k_

    def predict(self, X):
        """Anomaly labels.

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
            1 for anomalies (score above the threshold), 0 otherwise.
        """
        return (self.score_samples(X) > self.threshold_).astype(int)


class QuantumKernelClustering(_KernelMixin):
    """Spectral clustering on a kernel.

    The normalised affinity :math:`D^{-1/2} K D^{-1/2}` gives an embedding by its leading
    eigenvectors, which is clustered by k-means with several restarts.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters.
    kernel : {'quantum', 'rbf'}, optional
        Kernel; ``'rbf'`` is the classical baseline.
    reps, scale, gamma
        Kernel parameters, as in :class:`QuantumKernelAnomalyDetector`.
    restarts : int, optional
        k-means restarts.
    seed : int, optional
        Seed.
    """

    def __init__(self, n_clusters=2, kernel='quantum', reps=1, scale=0.5, gamma=0.5, restarts=10, seed=0):
        self.k, self.kernel, self.reps, self.scale, self.gamma = n_clusters, kernel, reps, scale, gamma
        self.restarts, self.seed = restarts, seed

    def fit_predict(self, X):
        """Cluster the samples.

        Parameters
        ----------
        X : array_like
            Samples.

        Returns
        -------
        numpy.ndarray
            Cluster label of each sample (also stored as ``labels_``).
        """
        X = self._kernel_fit(X)
        Kx = self.K(X)
        np.fill_diagonal(Kx, 0)
        d = Kx.sum(1)
        Dm = 1 / np.sqrt(np.maximum(d, 1e-12))
        w, V = np.linalg.eigh(Dm[:, None] * Kx * Dm[None, :])
        E = V[:, -self.k:]                                       # spectral embedding
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None)
        for _ in range(self.restarts):                          # k-means with random restarts
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
