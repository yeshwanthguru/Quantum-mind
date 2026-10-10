"""Quantum Perception–Dexterity Model (QPDM) research prototype.

QPDM exposes one sensor-sequence interface with interchangeable classical and
Qiskit statevector feature backends. The same classifier can be instantiated
independently for perception labels or dexterity primitives, then composed by
an application later. It is task-label generic: applications provide labels
and training examples; no handover or robot type is hard-coded.

This is a classical hybrid prototype, not evidence of quantum advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

SensorFrame = Mapping[str, float]
SensorSequence = Sequence[SensorFrame]


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]


class QuantumPerceptionDexterityModel:
    """Trainable sensor-sequence classifier with classical or Qiskit features.

    Args:
        labels: Application-defined perception labels or dexterity primitives.
        mode: Informational label, either "perception" or "dexterity".
        backend: "classical" or "qiskit". Qiskit is optional and uses an ideal
            local statevector circuit for the feature map.
        n_qubits: Number of qubits used by the Qiskit feature map.
        input_scales: Optional per-sensor characteristic scales. If omitted,
            scales are learned from the training set.
        learning_rate: SGD learning rate for the shared softmax readout.
        epochs: Number of readout training passes.
        seed: Deterministic initialization and training shuffle seed.

    Inputs are one mapping of named numeric readings or an ordered sequence of
    such mappings. For sequences, the encoder uses the latest reading, temporal
    mean, and first-to-last change. Missing known sensors are zero-filled.
    """

    def __init__(
        self,
        labels: Sequence[str],
        *,
        mode: str,
        backend: str = "classical",
        n_qubits: int = 4,
        input_scales: Mapping[str, float] | None = None,
        learning_rate: float = 0.05,
        epochs: int = 250,
        seed: int = 7,
    ) -> None:
        self.labels = tuple(labels)
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must contain at least two unique names")
        if mode not in {"perception", "dexterity"}:
            raise ValueError("mode must be 'perception' or 'dexterity'")
        if backend not in {"classical", "qiskit"}:
            raise ValueError("backend must be 'classical' or 'qiskit'")
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")
        if learning_rate <= 0 or epochs < 1:
            raise ValueError("learning_rate must be positive and epochs at least 1")
        self.mode = mode
        self.backend = backend
        self.n_qubits = int(n_qubits)
        self.input_scales = dict(input_scales or {})
        if any(not np.isfinite(v) or v <= 0 for v in self.input_scales.values()):
            raise ValueError("input scales must be finite and positive")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.sensor_names: tuple[str, ...] = ()
        self.scales: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    @staticmethod
    def _sequence(value: SensorFrame | SensorSequence) -> list[SensorFrame]:
        if isinstance(value, Mapping):
            frames = [value]
        else:
            frames = list(value)
        if not frames:
            raise ValueError("sensor sequence must contain at least one frame")
        for frame in frames:
            if not isinstance(frame, Mapping) or not frame:
                raise ValueError("each sensor frame must be a non-empty mapping")
            for name, reading in frame.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("sensor names must be non-empty strings")
                if not np.isfinite(float(reading)):
                    raise ValueError(f"sensor reading {name!r} must be finite")
        return frames

    @staticmethod
    def _raw_vector(value: SensorFrame | SensorSequence, names: tuple[str, ...]) -> np.ndarray:
        frames = QuantumPerceptionDexterityModel._sequence(value)
        data = np.asarray(
            [[float(frame.get(name, 0.0)) for name in names] for frame in frames],
            dtype=float,
        )
        latest = data[-1]
        mean = data.mean(axis=0)
        change = data[-1] - data[0]
        return np.concatenate((latest, mean, change))

    def _angles(self, value: SensorFrame | SensorSequence) -> np.ndarray:
        if self.scales is None:
            raise RuntimeError("call fit before prediction")
        raw = self._raw_vector(value, self.sensor_names)
        normalized = np.tanh(raw / self.scales)
        # Deterministically compress any sensor count to the configured circuit.
        folded = np.zeros(self.n_qubits, dtype=float)
        for index, reading in enumerate(normalized):
            folded[index % self.n_qubits] += reading / (1.0 + index // self.n_qubits)
        return np.pi * np.tanh(folded)

    def _features(self, value: SensorFrame | SensorSequence) -> np.ndarray:
        angles = self._angles(value)
        if self.backend == "classical":
            # Matched classical trigonometric feature map for the readout.
            return np.concatenate((np.sin(angles), np.cos(angles)))

        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp, Statevector
        except ImportError as exc:
            raise ImportError(
                "backend='qiskit' requires Qiskit; install qiskit to use this backend"
            ) from exc

        circuit = QuantumCircuit(self.n_qubits)
        for qubit, angle in enumerate(angles):
            circuit.ry(float(angle), qubit)
            circuit.rz(float(angle / 2.0), qubit)
        for qubit in range(self.n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        circuit.cx(self.n_qubits - 1, 0)
        state = Statevector.from_instruction(circuit)
        # Local Pauli-Z expectations are bounded real features for the common
        # trainable softmax readout. This prototype uses ideal statevector sim.
        return np.asarray(
            [float(np.real(state.expectation_value(
                SparsePauliOp.from_list(
                    [("I" * (self.n_qubits - 1 - q) + "Z" + "I" * q, 1.0)]
                )
            ))) for q in range(self.n_qubits)],
            dtype=float,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / values.sum(axis=1, keepdims=True)

    def fit(
        self,
        examples: Sequence[SensorFrame | SensorSequence],
        targets: Sequence[str],
    ) -> "QuantumPerceptionDexterityModel":
        """Fit the common supervised readout from application-labeled examples."""
        if len(examples) != len(targets) or len(examples) < 2:
            raise ValueError("examples and targets need equal length of at least two")
        unknown = set(targets) - set(self.labels)
        if unknown:
            raise ValueError(f"targets contain labels not configured: {sorted(unknown)}")
        frames = [self._sequence(example) for example in examples]
        names = sorted({name for sequence in frames for frame in sequence for name in frame})
        if not names:
            raise ValueError("training examples contain no sensor fields")
        self.sensor_names = tuple(names)
        raw = np.vstack([self._raw_vector(example, self.sensor_names) for example in examples])
        learned = np.maximum(np.max(np.abs(raw), axis=0), 1.0)
        self.scales = np.asarray(
            [self.input_scales.get(name, 1.0) for name in self.sensor_names] * 3,
            dtype=float,
        )
        # If no scale was supplied, use a robust per-feature training magnitude.
        for i, name in enumerate(self.sensor_names):
            if name not in self.input_scales:
                self.scales[i] = learned[i]
                self.scales[i + len(names)] = learned[i + len(names)]
                self.scales[i + 2 * len(names)] = learned[i + 2 * len(names)]

        x = np.vstack([self._features(example) for example in examples])
        y = np.asarray([self.labels.index(label) for label in targets], dtype=int)
        rng = np.random.default_rng(self.seed)
        self.weights = np.zeros((x.shape[1], len(self.labels)), dtype=float)
        self.bias = np.zeros(len(self.labels), dtype=float)
        for _ in range(self.epochs):
            for index in rng.permutation(len(x)):
                probabilities = self._softmax((x[index:index + 1] @ self.weights) + self.bias)[0]
                probabilities[y[index]] -= 1.0
                self.weights -= self.learning_rate * np.outer(x[index], probabilities)
                self.bias -= self.learning_rate * probabilities
        return self

    def predict_proba(self, value: SensorFrame | SensorSequence) -> dict[str, float]:
        """Return a normalized probability for each configured application label."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("call fit before prediction")
        features = self._features(value)
        probabilities = self._softmax(features[None, :] @ self.weights + self.bias)[0]
        return {label: float(probabilities[i]) for i, label in enumerate(self.labels)}

    def predict(self, value: SensorFrame | SensorSequence) -> Prediction:
        probabilities = self.predict_proba(value)
        label = max(probabilities, key=probabilities.get)
        return Prediction(label=label, probabilities=probabilities)


__all__ = ["Prediction", "QuantumPerceptionDexterityModel", "SensorFrame", "SensorSequence"]
