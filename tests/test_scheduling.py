"""Scheduling the quantum accelerometer with QUBO solvers: the exact objective, the surrogate, and the guarantees we can state."""
import numpy as np
import pytest

from quantum_mind.sensing import InterrogationProblem, random_mission, schedule_interrogations


def _mission(seed=1, n=10, k=3):
    return random_mission(seed, n_slots=n, k=k)


def test_exact_objective_matches_a_monte_carlo_of_the_real_filter():
    pr = _mission(0, n=12, k=4)
    sched = pr.optimum()
    mc = pr.monte_carlo(sched.slots, seeds=range(250))
    assert mc == pytest.approx(sched.objective, rel=0.10)


def test_more_readings_never_hurt_and_early_ones_help_most():
    pr = _mission(2, n=12, k=3)
    s = list(pr.feasible[:3])
    assert pr.objective(s + [int(pr.feasible[5])]) <= pr.objective(s) + 1e-12
    early, late = [int(pr.feasible[0])], [int(pr.feasible[-1])]
    assert pr.objective(early) < pr.objective(late)                  # the unknown MEMS bias is paid for from the start


def test_enumeration_is_optimal_and_every_method_respects_the_constraints():
    pr = _mission(1)
    opt = pr.optimum()
    candidates = [pr.uniform(), pr.greedy(), pr.random(0)]
    candidates += [schedule_interrogations(pr, m) for m in ('exact', 'annealing', 'greedy descent')]
    candidates.append(schedule_interrogations(pr, 'random sampling', shots=200))
    for c in candidates:
        assert len(set(c.slots)) == pr.k and set(c.slots) <= set(int(i) for i in pr.feasible)      # k distinct usable slots
        assert c.objective >= opt.objective - 1e-9                                                    # nothing beats the enumeration
        assert c.objective == pytest.approx(pr.objective(c.slots))


def test_the_local_quadratic_model_is_exact_within_two_flips_and_the_qubo_orders_schedules_alike():
    pr = _mission(3)
    f = [int(i) for i in pr.feasible]
    base = pr.uniform().slots
    lin, quad, const = pr.local_quadratic(base)

    def model(s):
        x = np.array([1.0 if i in s else 0.0 for i in f])
        return const + lin @ x + 0.5 * x @ quad @ x

    rng = np.random.default_rng(0)
    for _ in range(12):                                              # every schedule within two flips of the base is reproduced exactly
        s = set(base)
        for i in rng.choice(f, rng.integers(0, 3), replace=False):
            s ^= {int(i)}
        assert model(s) == pytest.approx(pr.variance(s), rel=1e-7, abs=1e-9)
    qp = pr.qubo(base)
    scale = np.abs(lin).max()
    a, b = set(base), set(base) - {base[0]} | {next(i for i in f if i not in base)}               # two schedules of size k, one swap apart
    xa = np.array([1.0 if i in a else 0.0 for i in f])
    xb = np.array([1.0 if i in b else 0.0 for i in f])
    assert (pr.variance(b) - pr.variance(a)) == pytest.approx(scale * (qp.objective(xb) - qp.objective(xa)), rel=1e-6, abs=1e-9)
    assert qp.feasible(xa) and qp.feasible(xb)


def test_iterating_from_the_empty_schedule_works_and_one_round_is_a_poorer_model():
    pr = _mission(0, n=14, k=4)
    opt = pr.optimum().objective
    one = schedule_interrogations(pr, 'exact', start='empty', rounds=1)
    many = schedule_interrogations(pr, 'exact', start='empty')
    assert len(set(one.slots)) == pr.k and one.info['rounds_used'] == 1                  # regression: it used to fall back
    assert one.slots != pr.uniform().slots
    assert many.objective <= one.objective + 1e-12 and many.objective <= 1.02 * opt


def test_the_iteration_never_ends_worse_than_its_start():
    for seed in range(3):
        pr = _mission(seed)
        for start in ('uniform', 'greedy'):
            res = schedule_interrogations(pr, 'exact', start=start)
            assert res.objective <= getattr(pr, start)().objective + 1e-12
            assert res.info['rounds_used'] >= 1


def test_the_qubo_methods_beat_random_and_the_polish_never_hurts():
    wins = 0
    for seed in range(4):
        pr = _mission(seed)
        plain = schedule_interrogations(pr, 'exact')
        polished = schedule_interrogations(pr, 'exact', polish=True)
        assert polished.objective <= plain.objective + 1e-12
        wins += plain.objective < np.mean([pr.random(s).objective for s in range(5)])
    assert wins >= 3


def test_inputs_are_validated():
    t, c = np.linspace(5, 50, 6), np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        InterrogationProblem(t, c, lambda x: np.ones_like(x), k=3, duration=60)          # only two usable slots
    with pytest.raises(ValueError):
        InterrogationProblem(t, np.ones(6), np.ones(7), k=2, duration=60)                  # wrong number of weights
