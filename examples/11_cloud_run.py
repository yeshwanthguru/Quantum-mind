"""Run any qlcog circuit on IBM Quantum or Amazon Braket hardware.

  python3 examples/11_cloud_run.py --backend ibm:least_busy --shots 4000
  python3 examples/11_cloud_run.py --backend braket:arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --shots 1000
  python3 examples/11_cloud_run.py --dry-run          # local simulators only, no account needed

IBM needs a saved Qiskit Runtime account; Braket needs AWS credentials and is billed per task and
shot. Results are printed with the analytic prediction; report hardware results as measured."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import argparse, json, datetime
from qlcog.core import tvd
from qlcog.circuits import run, order_effects_circuit
from qlcog.families.order_effects import QuantumOrderModel, qq_statistic

ap = argparse.ArgumentParser(); ap.add_argument('--backend', default='aer'); ap.add_argument('--shots', type=int, default=4000)
ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
backend = 'aer:FakeTorino' if a.dry_run else a.backend
form = 'deferred' if backend.startswith('braket') else 'dynamic'
m = QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2)); cells = {}
for order in ('AB', 'BA'):
    qc, dec = order_effects_circuit(m, order, form)
    cells[order] = dec(run(qc, backend, a.shots))
    print(order, 'circuit', cells[order].round(3), 'exact', m.predict()[order].round(3), 'TVD %.3f' % tvd(cells[order], m.predict()[order]))
print('QQ value from the circuit: %.4f (exact 0)' % qq_statistic(cells['AB'], cells['BA']))
print(json.dumps({'backend': backend, 'date': datetime.datetime.utcnow().isoformat(), 'cells': {k: v.tolist() for k, v in cells.items()}}))
