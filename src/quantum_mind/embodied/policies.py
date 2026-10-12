"""Closed-loop policies for the one-motor VLA: act on the argmax, hedge, or ask when the grounding is ambiguous."""
from __future__ import annotations

import numpy as np

from ..applications.calibration import SplitConformalClassifier
from .motor import OneMotor, expert_command
from .scene import cell_centres, clarification, encode, sample_scene


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


def vla_plan(model, gate=None, always_ask=False):
    """Plan factory. No gate: act on the model's expected target. With a gate: ask when the set is not a single cell."""
    xs = cell_centres(model.n_cells)

    def plan(scene, rng):
        g = model.ground(scene.image, scene.text)
        asked = 0
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
