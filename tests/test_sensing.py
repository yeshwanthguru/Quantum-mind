"""Quantum sensing: limits, sensor models and the dead-reckoning study."""
import numpy as np
import pytest

from quantum_mind.sensing import (Sensor, atom_interferometer_accelerometer, atom_interferometer_sensitivity,
                                  compare_sensor_suites, mems_accelerometer, phase_sensitivity, run_filter,
                                  simulate_dead_reckoning)


def test_quantum_limits():
    assert phase_sensitivity(100) == pytest.approx(0.1)
    assert phase_sensitivity(100, repeats=4) == pytest.approx(0.05)
    assert phase_sensitivity(100, scaling='heisenberg') == pytest.approx(0.01)
    assert phase_sensitivity(1e4, scaling='heisenberg') < phase_sensitivity(1e4)
    with pytest.raises(ValueError):
        phase_sensitivity(0.5)
    with pytest.raises(ValueError):
        phase_sensitivity(10, scaling='classical')


def test_atom_interferometer_sensitivity_scales_with_atoms_and_time():
    a = atom_interferometer_sensitivity(1e6, 0.1)
    assert atom_interferometer_sensitivity(4e6, 0.1) == pytest.approx(a / 2)         # 1/sqrt(N)
    assert atom_interferometer_sensitivity(1e6, 0.2) == pytest.approx(a / 4)         # 1/T^2
    assert atom_interferometer_sensitivity(1e6, 0.1, contrast=0.25) == pytest.approx(2 * a)
    assert atom_interferometer_sensitivity(1e6, 0.1, scaling='heisenberg') < a
    with pytest.raises(ValueError):
        atom_interferometer_sensitivity(1e6, 0.0)


def test_sensor_presets_have_the_expected_contrast():
    m, q = mems_accelerometer(), atom_interferometer_accelerometer()
    assert m.rate_hz > 10 * q.rate_hz                         # the MEMS sensor is fast
    assert q.noise_std < m.noise_std / 1e3                    # the quantum sensor is far more sensitive
    assert q.bias_walk < m.bias_walk                          # and almost drift-free
    with pytest.raises(ValueError):
        Sensor('bad', 0.0, 1.0, 0.0, 0.0)


def test_simulation_is_reproducible_and_filter_is_unbiased_at_start():
    m, q = mems_accelerometer(), atom_interferometer_accelerometer()
    d1 = simulate_dead_reckoning([m, q], duration=20, seed=3)
    d2 = simulate_dead_reckoning([m, q], duration=20, seed=3)
    assert np.allclose(d1.pos, d2.pos) and np.allclose(d1.readings[m.name], d2.readings[m.name])
    err = run_filter(d1)
    assert err[0] == 0.0 and np.all(np.isfinite(err))
    with pytest.raises(ValueError):
        run_filter(d1, use=['no such sensor'])


def test_fusing_a_stable_slow_sensor_beats_the_fast_drifting_one_alone():
    m, q = mems_accelerometer(), atom_interferometer_accelerometer()
    rows = compare_sensor_suites([m, q], {'mems': [m.name], 'fused': [m.name, q.name]}, duration=60, seeds=range(4))
    assert rows[1]['rmse'][0] < rows[0]['rmse'][0] / 3


def test_a_less_stable_quantum_sensor_gives_a_worse_fusion():
    m = mems_accelerometer()
    q_bad = atom_interferometer_accelerometer(bias_walk=m.bias_walk, bias_std0=m.bias_std0)
    q_good = atom_interferometer_accelerometer()
    suites = {'mems': [m.name], 'fused': [m.name, 'atom interferometer']}
    bad = compare_sensor_suites([m, q_bad], suites, duration=60, seeds=range(4))
    good = compare_sensor_suites([m, q_good], suites, duration=60, seeds=range(4))
    assert good[1]['rmse'][0] < bad[1]['rmse'][0]


def test_position_fixes_help_a_lot():
    m = mems_accelerometer()
    free = compare_sensor_suites([m], {'mems': [m.name]}, duration=60, seeds=range(3))
    fixed = compare_sensor_suites([m], {'mems': [m.name]}, duration=60, seeds=range(3), fix_period=5.0)
    assert fixed[0]['rmse'][0] < free[0]['rmse'][0] / 5
