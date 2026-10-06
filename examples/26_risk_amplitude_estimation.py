"""Finance (simulated): expected loss with quantum amplitude estimation versus classical Monte Carlo.

A loss distribution on 16 points (simulated) and a payoff scaled to [0, 1]. Maximum-likelihood
amplitude estimation (exact circuit simulation, sampled measurements) is compared with Monte Carlo
using the same number of calls to the distribution, over 50 repetitions."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.quantum import AmplitudeEstimation, monte_carlo_estimate

x = np.linspace(-3, 3, 16)
p = np.exp(-(x - 0.3) ** 2 / 1.6)                     # simulated loss distribution
f = np.clip((x + 3) / 6, 0, 1) ** 2                    # payoff in [0, 1]
ae = AmplitudeEstimation(p, f); rng = np.random.default_rng(0)
print('exact expected payoff %.5f' % ae.exact)
for powers in ((0,), (0, 1, 2, 4), (0, 1, 2, 4, 8, 16)):
    errs = {'ae': [], 'mc': []}
    for _ in range(50):
        r = ae.run(powers, shots=100, rng=rng)
        errs['ae'].append(r.estimate - r.exact); errs['mc'].append(monte_carlo_estimate(p, f, r.oracle_calls, rng) - r.exact)
    rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
    print('powers %-22s calls %5d   RMSE amplitude estimation %.5f   Monte Carlo %.5f'
          % (str(powers), r.oracle_calls, rmse['ae'], rmse['mc']))
try:
    from qiskit.quantum_info import Statevector
    sv = Statevector(ae.to_qiskit(2, measure=False))
    print('exported Qiskit circuit (m = 2): P(1) = %.5f, simulator %.5f' % (sv.probabilities([ae.n])[1], ae.good_probability(2)))
except ImportError:
    pass
