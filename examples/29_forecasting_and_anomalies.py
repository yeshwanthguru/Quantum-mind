"""Time series and fraud (simulated): forecasting with a variational quantum regressor, and anomaly
detection and clustering with the quantum kernel, each next to its classical baseline.

All data are simulated: a noisy seasonal signal for forecasting, and two-feature transaction records
with a small group of unusual ones for anomaly detection and clustering."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.quantum import VariationalRegressor, QuantumKernelAnomalyDetector, QuantumKernelClustering

rng = np.random.default_rng(0)
s = np.sin(np.arange(300) * 0.25) + 0.3 * np.sin(np.arange(300) * 0.07) + 0.05 * rng.normal(size=300)
W = 4; X = np.array([s[i:i + W] for i in range(len(s) - W)]); y = s[W:]; tr = 220
reg = VariationalRegressor(layers=3).fit(X[:tr], y[:tr])
coef = np.linalg.lstsq(np.c_[X[:tr], np.ones(tr)], y[:tr], rcond=None)[0]
lin = np.c_[X[tr:], np.ones(len(X) - tr)] @ coef
r2 = lambda pred: 1 - np.sum((pred - y[tr:]) ** 2) / np.sum((y[tr:] - y[tr:].mean()) ** 2)   # noqa: E731
print('forecasting R^2 on held-out data: variational quantum regressor %.3f, linear autoregression %.3f' % (reg.score(X[tr:], y[tr:]), r2(lin)))

normal = rng.normal(0, 1, (300, 2)); test = np.r_[rng.normal(0, 1, (150, 2)), rng.normal([3.2, -3.0], 0.4, (15, 2))]
label = np.r_[np.zeros(150), np.ones(15)]
for k in ('quantum', 'rbf'):
    det = QuantumKernelAnomalyDetector(kernel=k).fit(normal); sc = det.score_samples(test)
    auc = np.mean([sc[i] > sc[j] for i in np.flatnonzero(label) for j in np.flatnonzero(1 - label)])
    flagged = det.predict(test)
    print('anomaly detection (%-7s kernel): AUC %.3f, recall %.2f, false-positive rate %.3f'
          % (k, auc, flagged[label == 1].mean(), flagged[label == 0].mean()))
groups = np.r_[rng.normal([0, 0], 0.5, (80, 2)), rng.normal([2.5, 2.5], 0.5, (80, 2))]; truth = np.r_[np.zeros(80), np.ones(80)]
for k in ('quantum', 'rbf'):
    lab = QuantumKernelClustering(2, kernel=k).fit_predict(groups)
    print('clustering (%-7s kernel): accuracy %.3f' % (k, max(np.mean(lab == truth), np.mean(lab != truth))))
