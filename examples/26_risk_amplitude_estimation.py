"""Amplitude estimation versus Monte Carlo
=======================================

Finance (simulated): expected loss with quantum amplitude estimation versus classical Monte Carlo.

A loss distribution on 16 points (simulated) and a payoff scaled to [0, 1]. Maximum-likelihood
amplitude estimation (exact circuit simulation, sampled measurements) is compared with Monte Carlo
using the same number of calls to the distribution, over 50 repetitions.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/q.png'
import numpy as np
from quantum_mind.quantum import AmplitudeEstimation, monte_carlo_estimate

# %%
# A simulated loss distribution and payoff
# ----------------------------------------
x = np.linspace(-3, 3, 16)
p = np.exp(-(x - 0.3) ** 2 / 1.6)                     # simulated loss distribution
f = np.clip((x + 3) / 6, 0, 1) ** 2                    # payoff in [0, 1]
ae = AmplitudeEstimation(p, f); rng = np.random.default_rng(0)
print('exact expected payoff %.5f' % ae.exact)

# %%
# Error against the number of oracle calls
# ----------------------------------------
# Each setting is repeated 50 times; Monte Carlo gets as many samples as amplitude estimation makes
# calls to the state-preparation operator.
for powers in ((0,), (0, 1, 2, 4), (0, 1, 2, 4, 8, 16)):
    errs = {'ae': [], 'mc': []}
    for _ in range(50):
        r = ae.run(powers, shots=100, rng=rng)
        errs['ae'].append(r.estimate - r.exact); errs['mc'].append(monte_carlo_estimate(p, f, r.oracle_calls, rng) - r.exact)
    rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
    print('powers %-22s calls %5d   RMSE amplitude estimation %.5f   Monte Carlo %.5f'
          % (str(powers), r.oracle_calls, rmse['ae'], rmse['mc']))

# %%
# Check the exported Qiskit circuit
# ---------------------------------
try:
    from qiskit.quantum_info import Statevector
    sv = Statevector(ae.to_qiskit(2, measure=False))
    print('exported Qiskit circuit (m = 2): P(1) = %.5f, simulator %.5f' % (sv.probabilities([ae.n])[1], ae.good_probability(2)))
except ImportError:
    pass
