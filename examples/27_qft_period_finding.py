"""Signal processing: the quantum Fourier transform finds the period of a periodic state.

The circuit is checked against the discrete Fourier transform matrix, then applied to states with
period 4, 8 and 16 on 6 qubits (with and without measurement sampling)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.quantum import qft_circuit, qft_matrix, find_period

for n in (3, 4, 5):
    c = qft_circuit(n)
    U = np.column_stack([c.state(None, None, init=np.eye(2 ** n)[j])[0] for j in range(2 ** n)])
    print('QFT on %d qubits equals the DFT matrix: %s (%d gates)' % (n, np.allclose(U, qft_matrix(n)), len(c.ops)))
rng = np.random.default_rng(0)
for r in (4, 8, 16):
    P, exact = find_period(6, r, offset=1)
    _, sampled = find_period(6, r, offset=1, shots=200, rng=rng)
    print('true period %2d: from exact probabilities %s, from 200 samples %s; peaks at %s' % (r, exact, sampled, np.nonzero(P > 1e-9)[0].tolist()))
