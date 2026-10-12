"""Closed-loop policies for the one-motor VLA: act on the argmax, hedge, or ask when the grounding is ambiguous."""
from __future__ import annotations

import numpy as np

from ..applications.calibration import SplitConformalClassifier
from .motor import OneMotor, expert_command
from .belief import K, _restrict, person_answers
from .scene import cell_centres, clarification


def calibrate_gate(model, scenes, alpha=0.1):
    """Split-conformal gate fitted on held-out scenes: ask when the prediction set over cells has more than one cell."""
    g = model.ground(np.stack([s.image for s in scenes]), np.stack([s.text for s in scenes]))
    return SplitConformalClassifier(alpha=alpha).fit(g, np.array([s.target for s in scenes]))


def fuse(g_first, g_answer, eps=1e-6):
    """Classical belief update after an answer: product of the two groundings, renormalised (Bayes with independent evidence)."""
    post = g_first * g_answer
    return post / post.sum() if post.sum() > eps else g_answer


def run_episode(scene, rng, plan, motor=None, steps=60, tol=0.0625, n_cells=16):
    """Roll one episode out. ``plan(scene, rng)`` returns ``(controller, n_questions)``;
    ``controller(theta, omega)`` is called every control step with the noisy measurement."""
    motor = motor or OneMotor()
    xs = cell_centres(n_cells)
    controller, asked = plan(scene, rng)
    th, om = motor.reset(rng.uniform(-1, 1), rng)
    for _ in range(steps):
        th, om = motor.step(controller(th, om))
    goal = xs[scene.target]
    nearest = int(np.argmin([abs(motor.theta - xs[c]) for c in scene.cells]))
    return {'error': abs(motor.theta - goal), 'success': abs(motor.theta - goal) < tol,
            'right_object': scene.cells[nearest] == scene.target, 'asked': asked}


def oracle_plan(scene, rng):
    t = cell_centres(16)[scene.target]
    return (lambda th, om: float(expert_command(t, th, om))), 0


def vla_plan(model, gate=None, always_ask=False, slot=None, person=person_answers, n_questions=2):
    """Plan factory. No gate: act on the model's expected target. With a gate: ask when the set is not a single cell.

    With a belief ``slot`` the robot asks two questions, the simulated ``person`` answers them, and the slot (Bayes, context-weighted or
    quantum, see :mod:`quantum_mind.embodied.belief`) turns the instruction belief and the two answers into the posterior it acts on.
    ``n_questions=1`` uses only the first answer. The robot then acts on the most likely object. Without a slot it asks one
    noise-free question (an idealised answer) and fuses it by the product rule.
    """
    xs = cell_centres(model.n_cells)

    def plan(scene, rng):
        g = model.ground(scene.image, scene.text)
        asked = 0
        if slot is not None and (always_ask or (gate is not None and gate.predict_sets(g[None])[0].sum() > 1)):
            cells = sorted(scene.cells)
            m = len(cells)
            g0, L, ctx, mask = np.zeros((1, K)), np.zeros((1, 2, K)), np.zeros((1, 2), int), np.zeros((1, K))
            mask[0, :m], g0[0, :m] = 1, _restrict(g, cells)
            for j, (text, c) in enumerate(person(scene, rng)):
                L[0, j, :m], ctx[0, j] = _restrict(model.ground(scene.image, text), cells), c
                if j >= n_questions:
                    L[0, j, :m] = 1.0 / m                                  # unasked: uninformative
            post = slot.posterior(g0, L, ctx, mask)[0, :m]
            return (lambda th, om, t=float(xs[cells][int(np.argmax(post))]): float(model.act(t, th, om))), n_questions
        if always_ask or (gate is not None and gate.predict_sets(g[None])[0].sum() > 1):
            text2, _ = clarification(scene)
            g = fuse(g, model.ground(scene.image, text2))
            asked = 1
        t = float(g @ xs)
        return (lambda th, om: float(model.act(t, th, om))), asked
    return plan


def bc_plan(model):
    def plan(scene, rng):
        return (lambda th, om: model.command(scene.image, scene.text, th, om)), 0
    return plan


def run_sensed_episode(scene, rng, plan, mode, outage_at=1.0, disturbance=0.0, steps=60, tol=0.0625, n_cells=16):
    """Like :func:`run_episode`, but the controller sees a state *estimate* (see :mod:`quantum_mind.embodied.state`).

    The encoder drops out at ``outage_at`` seconds and an unmodelled push of up to ``disturbance`` workspace units/s^2 acts on the joint."""
    from .state import JointEstimator, SensedMotor
    controller, asked = plan(scene, rng)
    sm, est = SensedMotor(outage_at=outage_at, disturbance=disturbance), JointEstimator(mode)
    z0 = sm.reset(rng.uniform(-1, 1), rng)
    est.reset(z0)
    th, om = z0, 0.0
    for _ in range(steps):
        u = controller(th, om)
        th, om = est.step(sm.step(u), u)
    goal = cell_centres(n_cells)[scene.target]
    return {'error': abs(sm.m.theta - goal), 'success': abs(sm.m.theta - goal) < tol, 'asked': asked}
