"""Quantum-cognitive supervisory PID gain scheduler.

Research prototype only: recommends one of caller-supplied, pre-tuned PID
profiles. It does not calculate universal gains, command motor hardware, or
prove closed-loop stability. Validate profiles, their switching behavior, and
all operating envelopes for the target system before deployment.

CCQDF's internal risk value is a heuristic preference term, not a physical
safety monitor. Explicit motor limits and profile envelopes provide the guards.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

try:
    from .ccqdf import CCQDFModel
except ImportError:  # Allow importing when run from the repository root.
    from ccqdf import CCQDFModel


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and x >= 0.0 for x in (self.kp, self.ki, self.kd)):
            raise ValueError("PID gains must be finite and non-negative")


@dataclass(frozen=True)
class MotorContext:
    """Measured state in consistent, caller-defined units."""
    error: float
    error_rate: float
    speed: float
    load: float
    current: float
    temperature: float

    def __post_init__(self) -> None:
        values = (self.error, self.error_rate, self.speed, self.load, self.current, self.temperature)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("motor context values must all be finite")
        if self.current < 0.0 or self.temperature < 0.0:
            raise ValueError("current and temperature must be non-negative")


@dataclass(frozen=True)
class MotorSafetyLimits:
    """Hard limits; exceeding any one requests an emergency stop."""
    max_abs_speed: float
    max_current: float
    max_temperature: float

    def __post_init__(self) -> None:
        values = (self.max_abs_speed, self.max_current, self.max_temperature)
        if not all(math.isfinite(x) and x > 0.0 for x in values):
            raise ValueError("hard motor limits must be finite and positive")

    def exceeded_by(self, context: MotorContext) -> bool:
        return (
            abs(context.speed) > self.max_abs_speed
            or context.current > self.max_current
            or context.temperature > self.max_temperature
        )


@dataclass(frozen=True)
class OperatingEnvelope:
    """Validated state region in which a PID profile may be recommended."""
    max_abs_speed: float
    max_current: float
    max_temperature: float
    max_abs_load: float

    def __post_init__(self) -> None:
        values = (self.max_abs_speed, self.max_current, self.max_temperature, self.max_abs_load)
        if not all(math.isfinite(x) and x > 0.0 for x in values):
            raise ValueError("profile envelope limits must be finite and positive")

    def contains(self, context: MotorContext) -> bool:
        return (
            abs(context.speed) <= self.max_abs_speed
            and context.current <= self.max_current
            and context.temperature <= self.max_temperature
            and abs(context.load) <= self.max_abs_load
        )


@dataclass(frozen=True)
class PIDProfile:
    """Engineer-supplied gains, operating envelope, and relative preference."""
    name: str
    gains: PIDGains
    envelope: OperatingEnvelope
    preference: float = 0.5

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("profile name must be non-empty")
        if not math.isfinite(self.preference) or not 0.0 <= self.preference <= 1.0:
            raise ValueError("preference must be between 0 and 1")


@dataclass(frozen=True)
class PIDRecommendation:
    profile_name: str
    gains: PIDGains
    emergency_stop: bool
    eligible_profiles: tuple[str, ...]
    uncertainty: float | None
    reason: str


class QuantumCognitivePIDSupervisor:
    """Recommend among pre-tuned PID profiles using CCQDF context scoring.

    This class never writes to motor hardware. The caller must apply a
    recommendation through a controller with validated stability, saturation,
    anti-windup, and gain-transition handling.

    feature_scales supplies positive characteristic scales for exactly the six
    MotorContext fields. Each normalized input is passed through tanh and
    grouped into ordered context channels for CCQDF.
    """

    _FEATURES = ("error", "error_rate", "speed", "load", "current", "temperature")

    def __init__(
        self,
        profiles: Sequence[PIDProfile],
        *,
        safe_profile: str,
        hard_limits: MotorSafetyLimits,
        feature_scales: Mapping[str, float],
        minimum_dwell_s: float = 0.0,
        interference: float = 0.15,
        risk_weight: float = 0.5,
        temperature: float = 0.25,
        seed: int = 7,
    ) -> None:
        self.profiles = {p.name: p for p in profiles}
        if len(self.profiles) != len(profiles):
            raise ValueError("profile names must be unique")
        if len(self.profiles) < 2:
            raise ValueError("provide at least two profiles")
        if safe_profile not in self.profiles:
            raise ValueError("safe_profile must name a supplied profile")

        if set(feature_scales) != set(self._FEATURES):
            raise ValueError("feature_scales must define exactly all MotorContext fields")
        self.feature_scales = {k: float(v) for k, v in feature_scales.items()}
        if not all(math.isfinite(v) and v > 0.0 for v in self.feature_scales.values()):
            raise ValueError("feature scales must be finite and positive")
        if not math.isfinite(minimum_dwell_s) or minimum_dwell_s < 0.0:
            raise ValueError("minimum_dwell_s must be finite and non-negative")
        if not all(math.isfinite(v) and v >= 0.0 for v in (interference, risk_weight)):
            raise ValueError("interference and risk_weight must be finite and non-negative")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

        self.safe_profile = safe_profile
        self.hard_limits = hard_limits
        self.minimum_dwell_s = float(minimum_dwell_s)
        self.interference = float(interference)
        self.risk_weight = float(risk_weight)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self._models: dict[tuple[str, ...], CCQDFModel] = {}
        self._active_profile: str | None = None
        self._last_switch_s: float | None = None
        self._last_time_s: float | None = None

    def _context_channels(self, context: MotorContext) -> list[list[float]]:
        v = {
            name: math.tanh(getattr(context, name) / self.feature_scales[name])
            for name in self._FEATURES
        }
        return [
            [v["error"], v["error_rate"]],
            [v["speed"], v["load"]],
            [v["current"], v["temperature"]],
        ]

    def _model_for(self, names: tuple[str, ...]) -> CCQDFModel:
        if names not in self._models:
            offset = sum(
                (i + 1) * ord(char)
                for i, name in enumerate(names)
                for char in name
            )
            self._models[names] = CCQDFModel(
                actions=names,
                temperature=self.temperature,
                interference=self.interference,
                risk_weight=self.risk_weight,
                seed=self.seed + offset,
            )
        return self._models[names]

    def _recommend(
        self,
        name: str,
        *,
        stop: bool,
        eligible: Sequence[str],
        uncertainty: float | None,
        reason: str,
    ) -> PIDRecommendation:
        profile = self.profiles[name]
        return PIDRecommendation(
            profile_name=name,
            gains=profile.gains,
            emergency_stop=stop,
            eligible_profiles=tuple(eligible),
            uncertainty=uncertainty,
            reason=reason,
        )

    def recommend(self, context: MotorContext, *, now_s: float) -> PIDRecommendation:
        """Return a profile recommendation only; never send a motor command."""
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._last_time_s is not None and now_s < self._last_time_s:
            raise ValueError("now_s must be monotonic")
        self._last_time_s = float(now_s)

        safe = self.profiles[self.safe_profile]
        if self.hard_limits.exceeded_by(context):
            self._active_profile = safe.name
            self._last_switch_s = float(now_s)
            return self._recommend(
                safe.name, stop=True, eligible=(), uncertainty=None,
                reason="hard motor limit exceeded; stop requested and safe gains returned",
            )

        eligible = tuple(p.name for p in self.profiles.values() if p.envelope.contains(context))
        if not eligible:
            stop = not safe.envelope.contains(context)
            self._active_profile = safe.name
            self._last_switch_s = float(now_s)
            return self._recommend(
                safe.name, stop=stop, eligible=(), uncertainty=None,
                reason="no profile envelope contains this operating point; safe gains returned"
                + (" and stop requested" if stop else ""),
            )

        uncertainty: float | None = None
        try:
            if len(eligible) == 1:
                selected = eligible[0]
                reason = "only eligible pre-tuned PID profile selected"
            else:
                decision = self._model_for(eligible).decide(
                    self._context_channels(context),
                    utilities=[self.profiles[n].preference for n in eligible],
                    # Physical guards are separate: CCQDF risk is heuristic only.
                    risk_limits=[1.0] * len(eligible),
                )
                selected = decision.action
                uncertainty = decision.uncertainty
                reason = "CCQDF selected an eligible pre-tuned PID profile"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            stop = not safe.envelope.contains(context)
            self._active_profile = safe.name
            self._last_switch_s = float(now_s)
            return self._recommend(
                safe.name, stop=stop, eligible=eligible, uncertainty=None,
                reason=f"CCQDF scoring failed ({exc}); safe gains returned",
            )

        if (
            self._active_profile is not None
            and selected != self._active_profile
            and self._active_profile in eligible
            and self._last_switch_s is not None
            and now_s - self._last_switch_s < self.minimum_dwell_s
        ):
            selected = self._active_profile
            reason = "active eligible profile held during minimum dwell time"

        if selected != self._active_profile:
            self._active_profile = selected
            self._last_switch_s = float(now_s)

        return self._recommend(
            selected, stop=False, eligible=eligible,
            uncertainty=uncertainty, reason=reason,
        )


__all__ = [
    "MotorContext",
    "MotorSafetyLimits",
    "OperatingEnvelope",
    "PIDGains",
    "PIDProfile",
    "PIDRecommendation",
    "QuantumCognitivePIDSupervisor",
]
