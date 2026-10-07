"""Calibrated confidence for a robot's detector
=============================================

Robotics, perception: a detector's scores are over-confident (simulated). The example measures the
calibration error, recalibrates with Platt, temperature and isotonic scaling, builds conformal
prediction sets with a coverage guarantee, and turns a monitored lower bound into a hazard-aware
ask-or-act decision.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_32.png'
import numpy as np
from quantum_mind.applications.calibration import (expected_calibration_error, PlattScaling, TemperatureScaling,
                                                   IsotonicCalibration, SplitConformalClassifier, CalibrationMonitor)
from quantum_mind.applications.robotics import risk_aware_ask_or_act

# %%
# A simulated, over-confident five-class detector
# -----------------------------------------------
rng = np.random.default_rng(0)
logits = rng.normal(size=(6000, 5)) * 1.6
P_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
y = np.array([rng.choice(5, p=p) for p in P_true])
P = P_true ** 2.2 / (P_true ** 2.2).sum(1, keepdims=True)
conf, correct = P.max(1), P.argmax(1) == y
tr, te = slice(0, 3000), slice(3000, None)
print('raw ECE %.3f' % expected_calibration_error(conf[te], correct[te]))

# %%
# Recalibration on held-out data
# ------------------------------
for name, cal in (('Platt', PlattScaling()), ('isotonic', IsotonicCalibration())):
    print('%-12s ECE %.3f' % (name, expected_calibration_error(cal.fit(conf[tr], correct[tr]).transform(conf[te]), correct[te])))
ts = TemperatureScaling().fit(P[tr], y[tr])
print('temperature  ECE %.3f  (T = %.2f)' % (expected_calibration_error(ts.transform(P[te]).max(1), correct[te]), ts.T))

# %%
# Conformal sets and a monitored lower bound
# ------------------------------------------
S = SplitConformalClassifier(alpha=0.1).fit(P[tr], y[tr]).predict_sets(P[te])
print('conformal coverage %.3f (target 0.90), mean set size %.2f' % (S[np.arange(len(S)), y[te]].mean(), S.sum(1).mean()))
mon = CalibrationMonitor(min_count=20)
for c, ok in zip(conf[te], correct[te]):
    mon.update(c, ok)
est, lower = mon.estimate(0.95)
for hazard in ('low', 'high'):
    d = risk_aware_ask_or_act(0.95, hazard, p_lower=lower)
    print('detector says 0.95, observed %.2f, lower bound %.2f -> %-4s for a %s-hazard action' % (est, lower, d['action'], hazard))
