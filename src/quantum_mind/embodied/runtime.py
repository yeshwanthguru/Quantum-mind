"""Running the system on a robot: a hardware interface, an independent safety filter and a fixed-rate control loop.

Nothing in this module has been run on real hardware. :class:`SimulatedHardware` implements the interface on the simulator so
the loop, the safety filter and the failure handling are tested; a dev-kit driver has to implement :class:`MotorHardware` (three
methods) and be validated on the bench.

The safety filter (:class:`SafetyLayer`) sits between every command and the motor and does **not** depend on the learned model,
the agent, the supervisor or a language model: it clamps the command, limits its rate of change, refuses to drive past soft
position limits, and latches a stop that only a human releases. If the loop raises for any reason the runtime calls
``hardware.stop()`` before the exception leaves it.

Sensors the agent needs: a position encoder (workspace units, -1..1) and an accelerometer on the moving part (``a_mems``,
m/s^2). A drive with an encoder only cannot run :class:`~quantum_mind.embodied.agent.MotorAgent` as it is;
:class:`ControlRuntime` refuses to start rather than run degraded without saying so. The quantum accelerometer (``a_q``) is
optional, and no quantum accelerometer exists on a dev kit: it is simulated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .agent import MotorAgent
from .brain import Mode, Supervisor
from .state import SensedMotor


class MotorHardware(Protocol):
    """What a dev-kit driver must provide."""

    def read(self) -> dict:
        """``{'t': seconds, 'enc': position in -1..1 or None, 'a_mems': m/s^2 or None, 'a_q': m/s^2 or None}`` (latest values)."""

    def write(self, u: float) -> None:
        """Send a normalised command in [-1, 1] (the driver maps it to voltage, PWM or current)."""

    def stop(self) -> None:
        """Cut torque / apply the brake. Must be safe to call at any time, repeatedly."""


class SimulatedHardware:
    """The simulator behind the :class:`MotorHardware` interface."""

    def __init__(self, faults=(), seed=0, theta0=0.0, drop_telemetry_at=None, **kw):
        self.sm = SensedMotor(faults=faults, **kw)
        self.rng = np.random.default_rng(seed)
        self.sm.reset(theta0, self.rng)
        self._last = dict(t=0.0, enc=float(theta0), a_mems=0.0, a_q=None)
        self.stopped, self.drop_at, self._u = False, drop_telemetry_at, 0.0
        self.stop_calls = 0

    def read(self):
        return dict(self._last)

    def write(self, u):
        self._u = 0.0 if self.stopped else float(u)
        m = self.sm.step(self._u)
        if self.drop_at is None or self.sm.t < self.drop_at:
            self._last = dict(t=self.sm.t, enc=m['enc'], a_mems=m['a_mems'], a_q=m['a_q'])

    def stop(self):
        self.stopped = True
        self.stop_calls += 1

    @property
    def position(self):
        return self.sm.m.theta


@dataclass
class SafetyLimits:
    u_max: float = 0.8                 # fraction of full command allowed
    slew: float = 0.25                 # largest change of command per control step
    soft_limit: float = 0.95           # workspace position beyond which outward commands are refused


class SafetyLayer:
    """Last line of defence between any command and the motor; independent of the AI."""

    def __init__(self, limits=None):
        self.lim, self.u_prev, self.latched, self.log = limits or SafetyLimits(), 0.0, False, []

    def filter(self, u, position):
        if self.latched:
            self.u_prev = 0.0
            return 0.0
        u = float(np.clip(u, -self.lim.u_max, self.lim.u_max))
        u = float(np.clip(u, self.u_prev - self.lim.slew, self.u_prev + self.lim.slew))
        if position is not None and ((position > self.lim.soft_limit and u > 0) or (position < -self.lim.soft_limit and u < 0)):
            self.log.append('soft limit')
            u = 0.0
        self.u_prev = u
        return u

    def latch(self, reason):
        self.latched = True
        self.log.append('latched: ' + reason)

    def release(self):
        """Human action only."""
        self.latched, self.u_prev = False, 0.0


class ControlRuntime:
    """Fixed-rate loop: read, estimate, control, supervise, filter for safety, write. One :meth:`step` is one control period."""

    def __init__(self, hardware, act, brain=None, safety=None, heal=True, dt=0.05):
        self.hw, self.act, self.dt = hardware, act, dt
        self.agent, self.sup, self.safety = MotorAgent(heal=heal, dt=dt), Supervisor(brain=brain, dt=dt), safety or SafetyLayer()
        self.target, self.th, self.om, self.u, self.started = 0.0, 0.0, 0.0, 0.0, False

    def start(self):
        m = self.hw.read()
        if m.get('enc') is None or m.get('a_mems') is None:
            raise RuntimeError('the motor agent needs a position encoder and an accelerometer on the moving part; this hardware reports '
                               'enc=%r, a_mems=%r' % (m.get('enc'), m.get('a_mems')))
        self.agent.reset(m['enc'])
        self.th, self.om, self.started = m['enc'], 0.0, True

    def step(self, target=None):
        if target is not None:
            self.target = target
        try:
            if not self.started:
                self.start()
            u = self.sup.limit(self.agent.control(float(self.act(self.target, self.th, self.om))))
            u = self.safety.filter(u, self.th)
            if self.sup.mode == Mode.SAFE_STOP and not self.safety.latched:
                self.safety.latch('supervisor safe stop')
                self.hw.stop()
            self.hw.write(u)
            m = self.hw.read()
            fresh = m['t'] > getattr(self, '_t_prev', -1.0)
            self._t_prev = m['t']
            if fresh:
                self.th, self.om = self.agent.step(dict(enc=m['enc'], a_mems=m['a_mems'], a_q=m['a_q']), u)
            self.sup.observe(dict(t=self.sup.k * self.dt + self.dt, agent=self.agent, a_q=m['a_q'], fresh=fresh))
            self.u = u
        except BaseException:
            self.hw.stop()
            self.safety.latch('exception in the control loop')
            raise
        return self.sup.mode

    def run(self, steps, schedule=None):
        """Run ``steps`` periods; ``schedule`` maps a step index to a new target."""
        schedule = schedule or {}
        for k in range(steps):
            self.step(schedule.get(k))
        return self.sup.mode
