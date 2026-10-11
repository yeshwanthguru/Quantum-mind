"""Does a quantum accelerometer help a robot's dead reckoning?
==============================================================

Quantum sensing for robots, in simulation. A robot moves along a line with an unknown acceleration. A MEMS accelerometer is
fast but its bias drifts; a cold-atom interferometer is slow but quantum-limited and almost drift-free. A multi-rate Kalman filter
fuses them, with and without an occasional position fix. The sensor numbers are illustrative assumptions (the atom
interferometer's noise comes from the standard quantum limit); this is a simulation, not a navigation system.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_19.png'
from quantum_mind.sensing import (atom_interferometer_accelerometer, atom_interferometer_sensitivity, compare_sensor_suites,
                                  mems_accelerometer)

mems, quantum = mems_accelerometer(), atom_interferometer_accelerometer()
print('quantum limit for 1e6 atoms, T = 0.1 s: %.2e m/s^2 per shot' % atom_interferometer_sensitivity(1e6, 0.1))
for s in (mems, quantum):
    print('%-20s %6.2f Hz   white noise %.2e m/s^2   bias walk %.1e' % (s.name, s.rate_hz, s.noise_std, s.bias_walk))

# %%
# Position error with and without fixes
# -------------------------------------
suites = {'MEMS only': [mems.name], 'quantum only': [quantum.name], 'fused': [mems.name, quantum.name]}
for fix in (None, 10.0):
    print('\nposition fixes:', 'none' if fix is None else 'every %g s' % fix)
    for r in compare_sensor_suites([mems, quantum], suites, duration=120, seeds=range(4), fix_period=fix):
        print('  %-14s RMSE %.3f +- %.3f m' % ((r['suite'],) + r['rmse']))
