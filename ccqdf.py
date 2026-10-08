"""Contextual Counterfactual Quantum Decision Field (CCQDF).

CCQDF is a proposed quantum-cognitive decision model for robots. It is not a
claim that a robot or human brain performs physical quantum computation.

The central hypothesis is that a robot should select an action from the
counterfactual state produced by applying that action to a contextual cognitive
field, rather than from a static probability estimate alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class DecisionResult:
    """Result of one CCQDF decision."""
    action: str
    probabilities: dict[str, float]
    scores: dict[str, float]
    risk: dict[str, float]
    uncertainty: float
    counterfactual_gain: dict[str, float]
    reason: str

@dataclass(frozen=True)
class HandoverProfile:
    """Engineering-prior profile for the first robot use case."""
    actions: tuple[str, ...] = ("handover", "slow_handover", "ask", "wait")
    utility: tuple[float, ...] = (1.0, 0.82, 0.42, 0.10)
    risk_limit: tuple[float, ...] = (0.30, 0.42, 0.80, 0.95)

class CCQDFModel:
    """Contextual Counterfactual Quantum Decision Field.

    Context channels become Hermitian generators and therefore ordered unitary
    transformations. Candidate actions then receive their own counterfactual
    state transformation. Utility, phase-sensitive interference, risk and
    information value are combined before a hard safety gate and final choice.

    This is a classical numerical implementation of a quantum-cognitive
    representation; it does not require a quantum computer.
    """

    def __init__(self, actions: Sequence[str] | None = None, decay: float = 0.75,
                 temperature: float = 0.18, interference: float = 0.35,
                 risk_weight: float = 1.0, inertia: float = 0.12, seed: int = 7):
        self.actions = tuple(actions or HandoverProfile().actions)
        if len(self.actions) < 2 or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must contain at least two unique names")
        for name, value, lo, hi in (
            ("decay", decay, 0.0, 1.0),
            ("temperature", temperature, 1e-6, np.inf),
            ("interference", interference, 0.0, np.inf),
            ("risk_weight", risk_weight, 0.0, np.inf),
            ("inertia", inertia, 0.0, np.inf),
        ):
            if not lo <= value <= hi:
                raise ValueError(f"{name} is outside its valid range")
        self.decay = float(decay)
        self.temperature = float(temperature)
        self.interference = float(interference)
        self.risk_weight = float(risk_weight)
        self.inertia = float(inertia)
        rng = np.random.default_rng(seed)
        self._phase = rng.uniform(-np.pi, np.pi, len(self.actions))
        self._state = np.ones(len(self.actions), dtype=complex) / np.sqrt(len(self.actions))
        self._previous_action: str | None = None

    @staticmethod
    def _unitary(generator: np.ndarray) -> np.ndarray:
        """Return exp(-iG) for Hermitian G."""
        values, vectors = np.linalg.eigh(generator)
        return (vectors * np.exp(-1j * values)) @ vectors.conj().T

    def _context_generator(self, vector: np.ndarray, channel: int) -> np.ndarray:
        """Build a deterministic Hermitian generator from one context channel."""
        d = len(self.actions)
        x = np.asarray(vector, dtype=float).reshape(-1)
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("context channels must be finite non-empty vectors")
        z = np.resize(x, d)
        z = z / (np.linalg.norm(z) + 1e-12)
        shift = np.roll(z, channel % d)
        phase = np.exp(1j * (self._phase + channel * np.pi / 5.0))
        off = np.outer(z, shift) * phase[None, :]
        G = np.diag(z) + 0.5 * (off + off.conj().T)
        return (G + G.conj().T) / 2.0

    def _apply_context(self, state: np.ndarray, channels: Sequence[Sequence[float]]) -> np.ndarray:
        """Apply ordered context transformations to the cognitive field."""
        out = state.copy()
        for idx, channel in enumerate(channels):
            out = self._unitary(self._context_generator(np.asarray(channel), idx)) @ out
        return out / np.linalg.norm(out)

    def _counterfactual_state(self, state: np.ndarray, action_index: int) -> np.ndarray:
        """Generate the action-specific counterfactual field."""
        d = len(self.actions)
        e = np.zeros(d)
        e[action_index] = 1.0
        real_state = np.real(state)
        G = np.outer(e, real_state) + np.outer(real_state, e)
        G += np.diag(e * (1.0 + np.abs(state)))
        return self._unitary(0.45 * (G + G.T) / 2.0) @ state

    def _risk(self, probability: float, state: np.ndarray, action_index: int) -> float:
        """Estimate action risk from residual ambiguity and phase instability."""
        p = np.abs(state) ** 2
        entropy = -float(np.sum(np.where(p > 1e-12, p * np.log(np.clip(p, 1e-12, 1)), 0.0)))
        entropy /= np.log(len(self.actions))
        phase = abs(np.angle(state[action_index])) / np.pi
        return float(np.clip(0.65 * (1.0 - probability) + 0.25 * entropy + 0.10 * phase, 0.0, 1.0))

    def _ask_gain(self, state: np.ndarray, action_index: int) -> float:
        """Estimate the value of resolving ambiguity before an action."""
        p = np.abs(state) ** 2
        current = float(p[action_index])
        return (1.0 - current) ** 2 * (1.0 + np.std(p) * len(p))

    def decide(self, context: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
               utilities: Sequence[float] | None = None,
               risk_limits: Sequence[float] | None = None) -> DecisionResult:
        """Make one context-sensitive, counterfactual robot decision.

        Mapping insertion order is intentional: changing evidence order can
        change the field when context operators are non-commuting.
        """
        channels = list(context.values()) if isinstance(context, Mapping) else list(context)
        if not channels:
            raise ValueError("at least one context channel is required")
        state = self._apply_context(self._state, channels)
        counter = [self._counterfactual_state(state, i) for i in range(len(self.actions))]
        probs = np.array([abs(s[i]) ** 2 for i, s in enumerate(counter)], dtype=float)
        probs /= probs.sum()

        utilities = np.asarray(
            utilities if utilities is not None else np.linspace(1.0, 0.1, len(self.actions)), dtype=float
        )
        risk_limits = np.asarray(
            risk_limits if risk_limits is not None else np.full(len(self.actions), 0.5), dtype=float
        )
        if utilities.size != len(self.actions) or risk_limits.size != len(self.actions):
            raise ValueError("utilities and risk_limits must match the number of actions")
        if np.any(~np.isfinite(utilities)) or np.any((risk_limits < 0) | (risk_limits > 1)):
            raise ValueError("invalid utilities or risk limits")

        risks = np.array([self._risk(probs[i], counter[i], i) for i in range(len(self.actions))])
        gains = np.array([self._ask_gain(counter[i], i) for i in range(len(self.actions))])
        interference = np.array([
            np.real(state[i].conjugate() * counter[i][i]) for i in range(len(self.actions))
        ])
        scores = utilities * probs + self.interference * interference - self.risk_weight * risks
        scores += 0.25 * gains
        if self._previous_action in self.actions:
            scores -= self.inertia * (np.asarray(self.actions) != self._previous_action)

        unsafe = risks > risk_limits
        scores = np.where(unsafe, -np.inf, scores)
        if not np.any(np.isfinite(scores)):
            selected = self.actions.index("ask") if "ask" in self.actions else int(np.argmin(risks))
            reason = "all candidate actions exceeded their risk limits"
        else:
            logits = scores.copy()
            finite = np.isfinite(logits)
            logits[finite] = logits[finite] / self.temperature
            logits[finite] -= np.max(logits[finite])
            exp = np.zeros_like(logits)
            exp[finite] = np.exp(logits[finite])
            distribution = exp / exp.sum()
            selected = int(np.argmax(distribution))
            probs = distribution
            reason = "counterfactual utility-risk field selected the highest admissible action"

        uncertainty = float(-np.sum(np.where(probs > 0, probs * np.log2(np.clip(probs, 1e-12, 1)), 0.0)))
        self._previous_action = self.actions[selected]
        self._state = self.decay * self._state + (1.0 - self.decay) * state
        self._state /= np.linalg.norm(self._state)
        return DecisionResult(
            action=self.actions[selected],
            probabilities={a: float(probs[i]) for i, a in enumerate(self.actions)},
            scores={a: float(scores[i]) if np.isfinite(scores[i]) else float("-inf") for i, a in enumerate(self.actions)},
            risk={a: float(risks[i]) for i, a in enumerate(self.actions)},
            uncertainty=uncertainty,
            counterfactual_gain={a: float(gains[i]) for i, a in enumerate(self.actions)},
            reason=reason,
        )

    def reset(self) -> None:
        """Reset the temporal cognitive field."""
        self._state = np.ones(len(self.actions), dtype=complex) / np.sqrt(len(self.actions))
        self._previous_action = None

__all__ = ["CCQDFModel", "DecisionResult", "HandoverProfile"]
