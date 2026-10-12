"""A monitoring and self-healing agent for the motor.

Not a language-model agent: a supervisory loop with four jobs, run every control step.

1. **Estimate** (state slot): a Kalman filter fuses the encoder with the MEMS accelerometer and, when it is used, the slow quantum
   accelerometer, whose near-bias-free samples keep the MEMS bias calibrated.
2. **Monitor**: the accelerometer measures what the joint really did, so two redundancy checks are available. (a) The encoder is compared
   with a one-step dead-reckoned prediction from the accelerometer: a jump or a dropout is an encoder fault. (b) A single-fault
   least-squares fits of ``accel = gain*u - damping*omega - coulomb*tanh(omega/0.02) + push`` over the last second (one free parameter per
   hypothesis, so the fault types are not confounded).
3. **Diagnose**: an estimate that leaves the nominal band for several steps names the fault: actuator gain loss, friction increase,
   external push, encoder jump, encoder lost, or an unrecoverable actuator.
4. **Heal**: encoder faults mask the encoder and fall back on the accelerometers; plant faults re-linearise the command with the
   fitted plant (``u = (a_des + damping*omega + coulomb*tanh - push) / gain``); an unrecoverable actuator ends in a safe stop.

Only faults of those five kinds are handled. The agent knows the nominal plant (it is the controller's design model), which is the
classical assumption for fault detection; whether that holds on a real drive is a question for hardware.
"""
from __future__ import annotations

import numpy as np

from .state import JointEstimator

NOMINAL = np.array([8.0, 2.0, 0.3, 0.0])        # gain, damping, Coulomb friction, external acceleration (workspace units/s^2)


class MotorAgent:
    """Monitor, diagnose and heal one motor. ``heal=False`` only monitors; ``quantum=False`` leaves the quantum accelerometer out."""

    def __init__(self, heal=True, quantum=True, dt=0.05, stroke=0.2, window=16, persist=3):
        self.heal, self.dt, self.s, self.W, self.persist = heal, dt, stroke, window, persist
        self.mode = 'kalman + MEMS + quantum' if quantum else 'kalman + MEMS'
        self.quantum = quantum
        self.est = JointEstimator(self.mode, dt=dt, stroke=stroke)

    def reset(self, z0):
        self.est.reset(z0)
        self.est.quantum_bias = self.quantum
        self.ref = np.array(NOMINAL)                                   # healthy plant, calibrated online while nothing is wrong
        self.plant = np.array(NOMINAL)
        self.cal, self.Pcal = np.array(NOMINAL[:2]), np.diag([4.0, 1.0])
        self.buf = []
        self.t, self.enc_ok, self.bad_enc, self.none_enc, self.agree = 0.0, True, 0, 0, 0
        self.flags, self.counts, self.events, self.safe_stop, self.dead = set(), {}, [], False, 0
        self.label = None                                              # latest plant diagnosis; it can be revised as more data arrive
        self.theta_hat, self.omega_hat = z0, 0.0

    # ---- monitor ------------------------------------------------------------------------------------------------------
    def _log(self, kind):
        if kind not in self.flags:
            self.flags.add(kind)
            self.events.append((round(self.t, 2), kind))

    def step(self, meas, u):
        """Feed one set of measurements (and the command that produced them). Returns the state estimate ``(theta, omega)``."""
        self.t += self.dt
        x, dt = self.est.x, self.dt
        a_corr = meas['a_mems'] / self.s - x[2]                        # measured acceleration, bias removed
        enc = meas['enc']
        if enc is None:
            self.none_enc += 1
            if self.none_enc >= 2 and self.enc_ok:
                self.enc_ok = False
                self._log('encoder_lost')
        else:
            self.none_enc = 0
            pred = x[0] + x[1] * dt + 0.5 * dt * dt * a_corr           # dead-reckoned position
            jump = abs(enc - pred) > 0.035 + 0.004 * abs(a_corr)         # integrator mismatch grows with acceleration
            self.bad_enc = self.bad_enc + 1 if jump else 0
            if self.bad_enc >= 3 and self.enc_ok:
                self.enc_ok = False
                self._log('encoder_jump')
            if self.heal and not self.enc_ok:                          # masked: let the encoder back in once it agrees with the state again
                self.agree = self.agree + 1 if abs(enc - self.est.x[0]) < 0.04 else 0
                if self.agree >= 10:
                    self.enc_ok, self.agree, self.bad_enc = True, 0, 0
                    self.est.mode, self.est.plant = self.mode, None
                    self.events.append((round(self.t, 2), 'encoder_recovered'))
                    self.flags -= {'encoder_lost', 'encoder_jump'}
            if self.heal and (jump or not self.enc_ok):
                enc = None                                             # never feed a suspect reading to the filter
        if self.heal and not self.enc_ok:                              # heal: predict with the diagnosed plant, not the raw accelerometer
            self.est.mode, self.est.plant = 'kalman + motor model', self.plant
        theta, omega = self.est.step({**meas, 'enc': enc}, u)
        self.buf = (self.buf + [(a_corr, float(np.clip(u, -1, 1)), x[1])])[-self.W:]
        self._calibrate(a_corr, u, x[1])
        self._diagnose()
        self.theta_hat, self.omega_hat = theta, omega
        return theta, omega

    def _calibrate(self, y, u, omega):
        """Self-calibration: while the encoder is trusted and nothing is flagged, track gain and damping (recursive least squares)."""
        if not self.enc_ok or ({'actuator_degraded', 'friction_increase', 'external_push'} & self.flags):
            return
        phi = np.array([np.clip(u, -1, 1), -omega])
        Pphi = self.Pcal @ phi
        k = Pphi / (0.995 + phi @ Pphi + 0.05 ** 2)
        self.cal = self.cal + k * (y + NOMINAL[2] * np.tanh(omega / 0.02) - phi @ self.cal)
        self.Pcal = np.minimum((self.Pcal - np.outer(k, Pphi)) / 0.995, 20.0)
        self.ref = np.array([self.cal[0], self.cal[1], NOMINAL[2], 0.0])
        self.plant = self.ref

    def _fits(self):
        """Single-fault least-squares fits over the recent window: ``{fault: (plant, residual sum of squares)}``."""
        y, u, w = (np.array(c) for c in zip(*self.buf))
        g0, b0, c0, _ = self.ref
        th = np.tanh(w / 0.02)
        out = {}
        r0 = y - (g0 * u - b0 * w - c0 * th)
        out['none'] = (self.ref, float(r0 @ r0))
        if u @ u > 2.0:
            r = y + b0 * w + c0 * th
            g = float(u @ r / (u @ u))
            out['actuator_degraded'] = (np.array([g, b0, c0, 0.0]), float(((r - g * u) ** 2).sum()))
        d = float(r0.mean())
        out['external_push'] = (np.array([g0, b0, c0, d]), float(((r0 - d) ** 2).sum()))
        if th @ th > 2.0:
            r = y - g0 * u + b0 * w
            c = float(-th @ r / (th @ th))
            out['friction_increase'] = (np.array([g0, b0, c, 0.0]), float(((r + c * th) ** 2).sum()))
        return out

    def _joint(self):
        """Ridge fit of gain, Coulomb friction and push together, used to tell apart faults whose single-fault fits are close."""
        y, u, w = (np.array(c) for c in zip(*self.buf))
        g0, b0, c0, _ = self.ref
        th = np.tanh(w / 0.02)
        Phi = np.stack([u, -th, np.ones_like(u)], axis=1)
        r = y + b0 * w
        prior, lam = np.array([g0, c0, 0.0]), np.diag([1 / 3.0 ** 2, 1 / 0.7 ** 2, 1 / 1.0 ** 2])
        sol = np.linalg.solve(Phi.T @ Phi / 0.05 ** 2 + lam, Phi.T @ r / 0.05 ** 2 + lam @ prior)
        dev = {'actuator_degraded': (g0 - sol[0]) / (0.35 * g0), 'friction_increase': (sol[1] - c0) / 0.7,
               'external_push': abs(sol[2]) / 1.5}
        return dev, np.array([sol[0], b0, sol[1], sol[2]])

    def _diagnose(self):
        if len(self.buf) < self.W // 2:
            return
        fits = self._fits()
        rss0 = fits['none'][1]
        best, plant = 'none', self.ref
        for name, (p, rss) in fits.items():
            if name == 'none' or rss > 0.5 * rss0 or rss / len(self.buf) > 0.05:   # explain half the misfit AND fit the whole window well
                continue
            band = {'actuator_degraded': p[0] < 0.65 * self.ref[0], 'friction_increase': p[2] > 1.0, 'external_push': abs(p[3]) > 1.5}[name]
            if band and (best == 'none' or rss < fits[best][1]):
                best, plant = name, p
        close = [n for n, (p, rss) in fits.items() if n != 'none' and rss <= 0.5 * rss0 and rss / len(self.buf) <= 0.05
                 and rss <= 1.3 * fits[best][1]] if best != 'none' else []
        if len(close) > 1:                                             # ambiguous: let the joint fit decide which parameter moved most
            dev, joint = self._joint()
            pick = max(close, key=lambda n: dev[n])
            if dev[pick] > 1.0:
                best, plant = pick, joint
        plant_faults = ('actuator_degraded', 'friction_increase', 'external_push')
        self.counts = {k: (self.counts.get(k, 0) + 1 if k == best else 0) for k in plant_faults}
        if best != 'none' and self.counts[best] >= self.persist:
            self._log(best)
            self.label, self.plant = best, plant
        self.dead = self.dead + 1 if (best == 'actuator_degraded' and plant[0] < 1.2) else 0
        if self.dead >= 25:                                  # a transient mis-fit must not end in a latched safe stop
            self._log('actuator_unrecoverable')
            self.safe_stop = True

    # ---- heal ---------------------------------------------------------------------------------------------------------
    def control(self, u_pd):
        """Turn the controller's command into the one sent to the motor (identity unless healing and a plant fault is diagnosed)."""
        if not self.heal:
            return u_pd
        if self.safe_stop:
            return 0.0
        if not ({'actuator_degraded', 'friction_increase', 'external_push'} & self.flags):
            return u_pd
        g, b, c, d = self.plant
        w = self.omega_hat
        a_des = self.ref[0] * u_pd - self.ref[1] * w - self.ref[2] * np.tanh(w / 0.02)    # what the healthy loop would have done
        return float(np.clip((a_des + b * w + c * np.tanh(w / 0.02) - d) / max(g, 1.5), -1.0, 1.0))

    @property
    def final_diagnosis(self):
        """The fault the agent believes at the end: the encoder fault if one was found, else the latest plant diagnosis."""
        enc = [f for f in ('encoder_lost', 'encoder_jump') if f in self.flags]
        return enc[0] if enc else self.label

    @property
    def diagnosis(self):
        return sorted(self.flags)


def run_fault_episode(act, rng, faults=(), policy='agent', steps=80, switch_at=30, tol=0.0625):
    """Two-leg move with faults injected: go to target A, then (at ``switch_at``) to target B, which is where the score is taken.

    ``act(target, theta, omega)`` is the controller (for example ``TinyVLA.act``). ``policy`` is ``'encoder'`` (raw encoder, no agent),
    ``'kalman'`` (Kalman state estimate, no agent), ``'model'`` (Kalman with the nominal motor model, no agent), ``'monitor'`` (agent
    that only monitors), ``'agent no quantum'`` or ``'agent'``.
    """
    from .state import SensedMotor
    sm = SensedMotor(faults=faults)
    a, b = rng.uniform(-0.9, 0.9, 2)
    z0 = sm.reset(rng.uniform(-1, 1), rng)
    if policy in ('encoder', 'kalman', 'model'):
        modes = {'encoder': 'encoder', 'kalman': 'kalman + MEMS + quantum', 'model': 'kalman + motor model'}
        agent, est = None, JointEstimator(modes[policy])
        est.reset(z0)
    else:
        agent = MotorAgent(heal=policy != 'monitor', quantum=policy != 'agent no quantum')
        agent.reset(z0)
    th, om, u, peak = z0, 0.0, 0.0, 0.0
    for k in range(steps):
        tgt = a if k < switch_at else b
        u_pd = float(act(tgt, th, om))
        u = agent.control(u_pd) if agent else u_pd
        meas = sm.step(u)
        th, om = agent.step(meas, u) if agent else est.step(meas, u)
        if k >= switch_at:
            peak = max(peak, abs(sm.m.theta - b))
    err = abs(sm.m.theta - b)
    return {'error': err, 'success': err < tol, 'diagnosis': agent.diagnosis if agent else [],
            'events': agent.events if agent else [], 'final': agent.final_diagnosis if agent else None,
            'safe_stop': bool(agent and agent.safe_stop)}
