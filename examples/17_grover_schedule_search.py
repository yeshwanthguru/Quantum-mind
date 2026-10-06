"""Grover search for valid schedules
=================================

Scheduling: find the valid schedules among 2^5 options with Grover search.

Five binary choices (which of five maintenance slots to use). A schedule is valid when exactly two
slots are used and slots 0 and 1 are not both used. Grover amplifies the valid schedules; the circuit
is then sampled on the Aer simulator.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/q.png'
from qlcog.quantum import grover

# %%
# The validity rule as an oracle
# ------------------------------
valid = lambda x: sum(x) == 2 and not (x[0] and x[1])                 # noqa: E731
g = grover(5, valid)
print('%d valid schedules of 32; %d Grover iterations; P(valid) = %.3f (random guess %.3f)'
      % (len(g.marked), g.iterations, g.success, len(g.marked) / 32))

# %%
# Sample the circuit on Aer
# -------------------------
try:
    from qlcog.circuits import run
    counts = run(g.to_qiskit(), 'aer', 4000)
    hit = sum(k for b, k in counts.items() if valid(tuple(int(c) for c in b[::-1]))) / 4000
    print('Aer: fraction of valid samples %.3f' % hit)
except ImportError:
    print('install qiskit-aer to sample the circuit')
