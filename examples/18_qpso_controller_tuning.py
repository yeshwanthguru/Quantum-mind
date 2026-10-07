"""Tuning a PID controller with QPSO
=================================

Control engineering and robotics: tune a PID controller with quantum-behaved particle swarm
optimisation (quantum-inspired), compared with random search using the same number of evaluations.

Plant: a joint modelled as J q'' + b q' = u (illustrative parameters); cost: integrated squared error
of a unit step plus a control-effort penalty, simulated for 3 s.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_18.png'
import numpy as np
from quantum_mind.inspired import QPSO

# %%
# Plant and cost
# --------------
# Euler simulation of a unit step response; the cost adds a small penalty on control effort.
J, b, dt, T = 0.05, 0.2, 0.002, 3.0


def cost(gains):
    kp, ki, kd = gains; q = qd = integ = 0.0; e_prev = 1.0; c = 0.0
    for _ in range(int(T / dt)):
        e = 1.0 - q; integ += e * dt; u = kp * e + ki * integ + kd * (e - e_prev) / dt; e_prev = e
        u = np.clip(u, -20, 20); qdd = (u - b * qd) / J; qd += qdd * dt; q += qd * dt
        c += (e ** 2 + 1e-4 * u ** 2) * dt
        if abs(q) > 1e3:
            return 1e3
    return c


# %%
# QPSO versus random search
# -------------------------
# Both get the same number of cost evaluations.
lo, hi = [0, 0, 0], [50, 20, 5]
r = QPSO(cost, lo, hi, particles=16, iterations=40, seed=0).run()
rng = np.random.default_rng(0)
rs = min((cost(g), tuple(g)) for g in rng.uniform(lo, hi, (r.evaluations, 3)))
print('QPSO          gains kp=%.2f ki=%.2f kd=%.3f  cost %.4f  (%d evaluations)' % (*r.x, r.value, r.evaluations))
print('random search gains kp=%.2f ki=%.2f kd=%.3f  cost %.4f' % (*rs[1], rs[0]))
