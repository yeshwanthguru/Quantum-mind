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
from .policies import run_sensed_episode   # noqa: E402
from .state import MODES, JointEstimator, SensedMotor   # noqa: E402

__all__ += ['run_sensed_episode', 'JointEstimator', 'SensedMotor', 'MODES']
from .assignment import SOLVERS, assign_targets   # noqa: E402
from .system import QuantumRobotVLA   # noqa: E402

__all__ += ['assign_targets', 'SOLVERS', 'QuantumRobotVLA']
from .agent import MotorAgent, run_fault_episode   # noqa: E402

__all__ += ['MotorAgent', 'run_fault_episode']
from .brain import (ALLOWED, TRANSITIONS, Decision, Health, Heartbeat, LocalLLMBrain, Mode, MotorSubAgent, PerceptionSubAgent,   # noqa: E402
                    RuleBrain, SensorSubAgent, StateMachine, Status, Supervisor, ollama_generate, run_supervised_episode)

__all__ += ['Mode', 'Health', 'Status', 'Decision', 'StateMachine', 'TRANSITIONS', 'ALLOWED', 'MotorSubAgent', 'SensorSubAgent',
            'PerceptionSubAgent', 'Heartbeat', 'RuleBrain', 'LocalLLMBrain', 'ollama_generate', 'Supervisor', 'run_supervised_episode']
from .runtime import ControlRuntime, MotorHardware, SafetyLayer, SafetyLimits, SimulatedHardware   # noqa: E402

__all__ += ['MotorHardware', 'SimulatedHardware', 'SafetyLimits', 'SafetyLayer', 'ControlRuntime']
