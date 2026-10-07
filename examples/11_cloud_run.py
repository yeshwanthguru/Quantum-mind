"""Running on IBM Quantum or Amazon Braket hardware
================================================

Run any quantum_mind circuit on IBM Quantum or Amazon Braket hardware.

  python3 examples/11_cloud_run.py --backend ibm:least_busy --shots 4000
  python3 examples/11_cloud_run.py --backend braket:arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --shots 1000
  python3 examples/11_cloud_run.py --dry-run          # local simulators only, no account needed

IBM needs a saved Qiskit Runtime account; Braket needs AWS credentials and is billed per task and
shot. Results are printed with the analytic prediction; report hardware results as measured.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_11.png'
import argparse, json, datetime
from quantum_mind.core import tvd
from quantum_mind.circuits import run, order_effects_circuit
from quantum_mind.families.order_effects import QuantumOrderModel, qq_statistic

# %%
# Command-line options
# --------------------
# The back end and the number of shots; ``--dry-run`` uses the FakeTorino noise model instead of an
# account. Braket devices need the deferred-measurement form of the circuit.
ap = argparse.ArgumentParser(); ap.add_argument('--backend', default='aer'); ap.add_argument('--shots', type=int, default=4000)
ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
backend = 'aer:FakeTorino' if a.dry_run else a.backend
form = 'deferred' if backend.startswith('braket') else 'dynamic'
m = QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2)); cells = {}

# %%
# Run both question orders
# ------------------------
# Each circuit is decoded into answer probabilities and compared with the exact model. The QQ value
# should be close to 0 on any device; a large value means noise or an error.
for order in ('AB', 'BA'):
    qc, dec = order_effects_circuit(m, order, form)
    cells[order] = dec(run(qc, backend, a.shots))
    print(order, 'circuit', cells[order].round(3), 'exact', m.predict()[order].round(3), 'TVD %.3f' % tvd(cells[order], m.predict()[order]))
print('QQ value from the circuit: %.4f (exact 0)' % qq_statistic(cells['AB'], cells['BA']))
print(json.dumps({'backend': backend, 'date': datetime.datetime.utcnow().isoformat(), 'cells': {k: v.tolist() for k, v in cells.items()}}))
