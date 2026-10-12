"""Embodied AI: a tiny vision-language-action (VLA) prototype for one motor, as a framework of replaceable slots.

This is the prototype of the fourth strand of the library. It does not compete with large VLA models; it is a small,
fully inspectable system on which the library's own ideas (calibrated grounding, conformal ask-or-act, later quantum
beliefs and quantum-sensor state) can be tested against honest baselines before anything is scaled.

* :class:`OneMotor`: simulated DC-motor joint with a noisy encoder; :func:`expert_command` is the PD demonstrator.
* :func:`sample_scene`, :func:`clarification`: a one-dimensional tabletop, instructions, and clarifying answers.
* :class:`TinyVLA`: attention grounding plus an action head, trained end to end in NumPy with hand-written gradients.
* :class:`BCBaseline`: the usual end-to-end behaviour-cloning policy, the baseline.
* :func:`vla_plan`, :func:`bc_plan`, :func:`oracle_plan`, :func:`calibrate_gate`, :func:`run_episode`: closed-loop evaluation.
"""
from .motor import OneMotor, expert_command
from .policies import bc_plan, calibrate_gate, fuse, oracle_plan, run_episode, vla_plan
from .scene import COLORS, WORDS, Scene, cell_centres, clarification, sample_scene
from .vla import BCBaseline, TinyVLA, make_dataset

__all__ = ['OneMotor', 'expert_command', 'Scene', 'sample_scene', 'clarification', 'cell_centres', 'WORDS', 'COLORS',
           'TinyVLA', 'BCBaseline', 'make_dataset', 'vla_plan', 'bc_plan', 'oracle_plan', 'calibrate_gate', 'fuse',
           'run_episode']
from .belief import BayesSlot, QuantumSlot, TemperedSlot, person_answers, sequence_batch   # noqa: E402

__all__ += ['BayesSlot', 'TemperedSlot', 'QuantumSlot', 'person_answers', 'sequence_batch']
