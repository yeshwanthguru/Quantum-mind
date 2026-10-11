"""Quantum sensing for robots: sensor models with their quantum limits, and fusion with classical sensors.

This is the second of the three threads of quantum robotics that Quantum Mind covers (the others are quantum computing for
robot algorithms, :mod:`quantum_mind.computing`, and quantum-probability beliefs about people, :mod:`quantum_mind.belief`).
Quantum sensors are physical instruments, so this module does not build one. It lets you ask, in simulation and with the
sensor's numbers, whether a slow, almost drift-free quantum sensor changes a robot's position estimate and when it does not.

* :func:`phase_sensitivity`, :func:`atom_interferometer_sensitivity`: the standard quantum limit and the acceleration
  sensitivity of an atom interferometer.
* :class:`Sensor`, :func:`mems_accelerometer`, :func:`atom_interferometer_accelerometer`: accelerometer models.
* :func:`simulate_dead_reckoning`, :func:`run_filter`, :func:`compare_sensor_suites`: a one-axis dead-reckoning study with a
  multi-rate Kalman filter.
"""
from .dead_reckoning import DeadReckoningData, compare_sensor_suites, run_filter, simulate_dead_reckoning
from .limits import RB87_WAVELENGTH_M, atom_interferometer_sensitivity, phase_sensitivity
from .sensors import Sensor, atom_interferometer_accelerometer, mems_accelerometer

__all__ = ['DeadReckoningData', 'Sensor', 'compare_sensor_suites', 'run_filter', 'simulate_dead_reckoning',
           'phase_sensitivity', 'atom_interferometer_sensitivity', 'RB87_WAVELENGTH_M', 'mems_accelerometer',
           'atom_interferometer_accelerometer']
