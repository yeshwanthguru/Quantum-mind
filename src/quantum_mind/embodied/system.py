"""The whole system in one place: a VLA whose belief, state and assignment slots come from the three strands of quantum robotics."""
from __future__ import annotations

from dataclasses import dataclass, field

from .assignment import assign_targets
from .belief import BayesSlot, person_answers
from .policies import calibrate_gate, run_sensed_episode, vla_plan
from .vla import TinyVLA


@dataclass
class QuantumRobotVLA:
    """Quantum-robotics VLA (QR-VLA): one grounding model and three slots.

    * grounding: :class:`TinyVLA` (classical attention over the image, conditioned on the instruction);
    * belief (quantum cognition): ``slot``, e.g. :class:`~quantum_mind.embodied.belief.QuantumSlot`, fuses the instruction with the answers;
    * state (quantum sensing): ``state_mode`` of :class:`~quantum_mind.embodied.state.JointEstimator`, e.g. ``'kalman + MEMS + quantum'``;
    * assignment (quantum computing): :func:`~quantum_mind.embodied.assignment.assign_targets` with the chosen solver.

    The quantum parts are simulated on a classical machine, and each has a classical alternative that is the baseline for it.
    """
    vla: TinyVLA
    gate: object = None
    slot: object = field(default_factory=BayesSlot)
    state_mode: str = 'kalman + MEMS + quantum'
    solver: str = 'exact'

    def plan(self, person=person_answers, n_questions=2):
        """Closed-loop plan: ground, ask through the conformal gate when ambiguous, fuse the answers with the belief slot."""
        return vla_plan(self.vla, self.gate, slot=self.slot, person=person, n_questions=n_questions)

    def run(self, scene, rng, **kw):
        """One episode with the state slot between the sensors and the controller."""
        return run_sensed_episode(scene, rng, self.plan(), self.state_mode, **kw)

    def assign(self, motor_positions, target_positions, **kw):
        """Which motor goes to which target (computing strand)."""
        return assign_targets(motor_positions, target_positions, self.solver, **kw)

    @classmethod
    def fit(cls, scenes, calibration_scenes, slot=None, seed=0, steps=3000, alpha=0.02, **kw):
        vla = TinyVLA(seed=seed).fit(scenes, steps=steps, seed=seed)
        return cls(vla, calibrate_gate(vla, calibration_scenes, alpha), slot or BayesSlot(), **kw)
