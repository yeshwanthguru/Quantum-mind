"""PyBullet simulation of trust-aware hand-overs, recorded as a GIF.

A KUKA iiwa arm picks a cube from a stand and hands it to a simulated person. Before each hand-over
the robot's policy (:mod:`quantum_handover.policies`) chooses to hand over, hand over slowly, ask
"Are you ready?" or wait. Whether the person takes the cube is drawn from the person's true trust;
when they do not, the arm releases the cube and it falls under gravity. The arm moves kinematically
(joint positions from inverse kinematics); the falling cube is simulated by the physics engine.

The decisions and outcomes follow exactly the same rules and random draws as
:func:`quantum_mind.applications.handover.simulate_handover_session`, so the animation shows one of
the sessions the benchmark counts. Requires ``pybullet`` and ``pillow`` (``pip install
"quantum-handover[sim]"``). Rendering uses PyBullet's CPU renderer, so no display or GPU is needed.
"""
from __future__ import annotations

import numpy as np

__all__ = ['HandoverScene', 'run_session', 'side_by_side', 'save_gif']

PICK = np.array([0.45, -0.45, 0.33])        # cube position on the stand
HAND = np.array([0.45, 0.45, 0.62])         # where the person takes the cube
DOWN = None                                  # end-effector orientation pointing down (set on connect)


class HandoverScene:
    """The robot, the stand, the person and the cube, with an off-screen camera.

    Parameters
    ----------
    width, height : int, optional
        Image size of every frame.
    """

    def __init__(self, width=320, height=240):
        import pybullet as p
        import pybullet_data
        self.p, self.w, self.h = p, width, height
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF('plane.urdf', physicsClientId=self.cid)
        self.robot = p.loadURDF('kuka_iiwa/model.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=self.cid)
        self.ee = p.getNumJoints(self.robot, physicsClientId=self.cid) - 1
        self.down = p.getQuaternionFromEuler([0, np.pi, 0])
        box = lambda half, rgba, pos: p.createMultiBody(                         # noqa: E731
            0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid),
            p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=self.cid),
            pos, physicsClientId=self.cid)
        box([0.08, 0.08, 0.15], [0.55, 0.55, 0.6, 1], [PICK[0], PICK[1], 0.15])           # stand
        sphere = lambda r, rgba, pos: p.createMultiBody(                         # noqa: E731
            0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba, physicsClientId=self.cid), pos,
            physicsClientId=self.cid)
        torso = p.createVisualShape(p.GEOM_CAPSULE, radius=0.16, length=0.55, rgbaColor=[0.35, 0.45, 0.75, 1],
                                    physicsClientId=self.cid)
        p.createMultiBody(0, -1, torso, [HAND[0] + 0.05, HAND[1] + 0.45, 0.62], physicsClientId=self.cid)
        self.head = sphere(0.12, [0.93, 0.78, 0.65, 1], [HAND[0] + 0.05, HAND[1] + 0.45, 1.12])
        self.hand = sphere(0.05, [0.93, 0.78, 0.65, 1], HAND)
        self.cube = p.loadURDF('cube_small.urdf', PICK + [0, 0, 0.03], physicsClientId=self.cid)
        p.changeVisualShape(self.cube, -1, rgbaColor=[0.95, 0.6, 0.15, 1], physicsClientId=self.cid)
        self.view = p.computeViewMatrixFromYawPitchRoll([0.35, 0.1, 0.5], 2.0, 45, -24, 0, 2, physicsClientId=self.cid)
        self.proj = p.computeProjectionMatrixFOV(55, width / height, 0.05, 5, physicsClientId=self.cid)
        self.q = self.ik(PICK + [0, 0, 0.25])
        self.set_joints(self.q)

    # ------------------------------------------------------------------ helpers
    def ik(self, target):
        """Joint positions that put the end effector at ``target``, pointing down."""
        q = self.p.calculateInverseKinematics(self.robot, self.ee, target, self.down, maxNumIterations=200,
                                              physicsClientId=self.cid)
        return np.array(q[:7])

    def set_joints(self, q):
        for j, v in enumerate(q):
            self.p.resetJointState(self.robot, j, float(v), physicsClientId=self.cid)

    def ee_pos(self):
        return np.array(self.p.getLinkState(self.robot, self.ee, physicsClientId=self.cid)[4])

    def place_cube(self, pos):
        self.p.resetBasePositionAndOrientation(self.cube, list(pos), [0, 0, 0, 1], physicsClientId=self.cid)
        self.p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def colour_hand(self, rgba):
        self.p.changeVisualShape(self.hand, -1, rgbaColor=rgba, physicsClientId=self.cid)

    def frame(self):
        """RGB image of the current scene."""
        _, _, rgba, _, _ = self.p.getCameraImage(self.w, self.h, self.view, self.proj,
                                                 renderer=self.p.ER_TINY_RENDERER, physicsClientId=self.cid)
        return np.asarray(rgba, np.uint8).reshape(self.h, self.w, 4)[:, :, :3]

    def close(self):
        self.p.disconnect(physicsClientId=self.cid)


def _caption(img, lines):
    """Draw caption lines on a frame (top-left)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:                                    # Pillow < 10.1
        font = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 16 * len(lines) + 6], fill=(13, 17, 23))
    for i, text in enumerate(lines):
        d.text((6, 4 + 16 * i), text, fill=(230, 237, 243), font=font)
    return np.asarray(im)


def run_session(policy, person, rng, n_steps=10, width=320, height=240, title=None):
    """Simulate one session and record its frames.

    Parameters
    ----------
    policy : TrustAwareHandover
        The robot's policy (see :func:`quantum_handover.policies.make_policy`).
    person : TrustAwareHandover
        The person's true trust state (see :func:`quantum_handover.policies.person_state`).
    rng : numpy.random.Generator
        Random draws for answers and outcomes, in the same order as
        :func:`quantum_mind.applications.handover.simulate_handover_session`.
    n_steps : int, optional
    width, height : int, optional
    title : str, optional
        First caption line of every frame, for example the policy's name.

    Returns
    -------
    frames : list of numpy.ndarray
    log : list of dict
        One entry per step: ``action``, ``outcome`` (``'taken'``, ``'dropped'``, ``'yes'``, ``'no'`` or
        ``'waited'``), ``robot_p_trust``, ``person_p_trust`` and ``cost`` so far.
    """
    scene = HandoverScene(width, height)
    frames, log, total = [], [], 0.0
    skin, green, red = [0.93, 0.78, 0.65, 1], [0.25, 0.8, 0.35, 1], [0.9, 0.25, 0.2, 1]

    def shoot(caption, k=1):
        img = _caption(scene.frame(), ([title] if title else []) + list(caption))
        frames.extend([img] * k)

    def move(target, n, caption, carry=False):
        q1 = scene.ik(target)
        for s in np.linspace(0, 1, n + 1)[1:]:
            scene.set_joints((1 - s) * scene.q + s * q1)
            if carry:
                scene.place_cube(scene.ee_pos() - [0, 0, 0.05])
            shoot(caption)
        scene.q = q1

    try:
        for step in range(n_steps):
            d = policy.decide()
            a = d['action']
            head = ['step %d/%d  robot: %s' % (step + 1, n_steps, a.replace('_', ' ')),
                    'P(trust) robot believes %.2f | person %.2f | cost %.1f' % (policy.p_trust, person.p_trust, total)]
            if a == 'ask':
                total += policy.ask_cost
                yes = rng.random() < person.p_trust
                person.answer(yes)
                policy.answer(yes)
                move(HAND + [0, -0.18, 0.12], 3, head)
                shoot(head[:1] + ['"Are you ready?"  person: %s' % ('yes' if yes else 'no')], 6)
                log.append({'action': a, 'outcome': 'yes' if yes else 'no'})
            elif a == 'wait':
                total += policy.wait_cost
                person.wait()
                policy.wait()
                shoot(head, 5)
                log.append({'action': a, 'outcome': 'waited'})
            else:
                slow = a == 'slow_handover'
                p_fail = (1 - person.p_trust) * (policy.slow_factor if slow else 1.0)
                if slow:
                    total += policy.slow_cost
                ok = rng.random() >= p_fail
                total += 0 if ok else policy.fail_cost
                person.observe(int(ok))
                policy.observe(int(ok))
                scene.place_cube(PICK + [0, 0, 0.03])
                move(PICK + [0, 0, 0.12], 3, head)
                move(PICK + [0, 0, 0.07], 2, head)
                move(PICK + [0, 0, 0.25], 2, head, carry=True)
                move(HAND + [0, 0, 0.06], 10 if slow else 5, head, carry=True)
                if ok:
                    scene.colour_hand(green)
                    scene.place_cube(HAND + [0, 0, 0.05])
                    shoot(head[:1] + ['person takes the cube'], 5)
                else:
                    scene.colour_hand(red)
                    for _ in range(6):                    # the cube falls under gravity
                        for _ in range(12):
                            scene.p.stepSimulation(physicsClientId=scene.cid)
                        shoot(head[:1] + ['hand-over failed: the cube drops'])
                scene.colour_hand(skin)
                move(PICK + [0, 0, 0.25], 3, head)
                log.append({'action': a, 'outcome': 'taken' if ok else 'dropped'})
            log[-1].update(robot_p_trust=policy.p_trust, person_p_trust=person.p_trust, cost=total)
        shoot(['session over: total cost %.1f' % total,
               '%d hand-overs, %d dropped, %d questions' % (
                   sum(e['outcome'] in ('taken', 'dropped') for e in log),
                   sum(e['outcome'] == 'dropped' for e in log), sum(e['action'] == 'ask' for e in log))], 15)
    finally:
        scene.close()
    return frames, log


def side_by_side(left, right, gap=4):
    """Join two frame sequences side by side; the shorter one holds its last frame."""
    n = max(len(left), len(right))
    pad = lambda fs: fs + [fs[-1]] * (n - len(fs))                         # noqa: E731
    bar = np.full((left[0].shape[0], gap, 3), 13, np.uint8)
    return [np.hstack([a, bar, b]) for a, b in zip(pad(left), pad(right))]


def save_gif(frames, path, ms=70):
    """Write frames as an animated GIF (adaptive palette, looping)."""
    from PIL import Image
    ims = [Image.fromarray(f).convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    return path
