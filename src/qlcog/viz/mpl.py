"""Matplotlib Bloch spheres: static figures, animations and GIF/MP4 export."""
from __future__ import annotations

import numpy as np
from .themes import theme as _theme, AXIS_LABELS
from .states import Trajectory, bloch_vectors

__all__ = ['BlochSphere', 'plot_bloch', 'animate_trajectory']


class BlochSphere:
    """One Bloch sphere on a Matplotlib 3D axis.

        b = BlochSphere(title='q0'); b.add_vector([0, 1, 0]); b.add_trajectory(points); b.show()"""

    def __init__(self, ax=None, title='', theme='dark', figsize=(5, 5), elev=20, azim=35):
        import matplotlib.pyplot as plt
        self.t = _theme(theme)
        if ax is None:
            self.fig = plt.figure(figsize=figsize, facecolor=self.t['bg'])
            ax = self.fig.add_subplot(projection='3d')
        else:
            self.fig = ax.figure
        self.ax = ax; self.title = title; self._n = 0
        self._draw_frame(elev, azim)

    def _draw_frame(self, elev, azim):
        ax, t = self.ax, self.t
        ax.set_facecolor(t['bg'])
        u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
        ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=t['sphere'],
                        alpha=t['sphere_alpha'], linewidth=0, shade=False)
        th = np.linspace(0, 2 * np.pi, 120)
        for k in (-0.5, 0.0, 0.5):                                    # latitudes
            r = np.sqrt(1 - k ** 2)
            ax.plot(r * np.cos(th), r * np.sin(th), k, color=t['wire'], lw=0.8 if k else 1.2)
        for phi in np.linspace(0, np.pi, 4, endpoint=False):           # meridians
            ax.plot(np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), np.sin(th), color=t['wire'], lw=0.8)
        for a in np.eye(3):
            ax.plot(*np.c_[-a, a], color=t['axis'], lw=0.8, ls='--')
        for pos, lab in AXIS_LABELS.items():
            ax.text(*pos, lab, color=t['text'], fontsize=11, ha='center', va='center')
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1), zoom=1.22); ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
        if self.title:
            ax.text2D(0.5, 0.0, self.title, transform=ax.transAxes, color=t["text"], fontsize=12, ha="center")

    def _color(self, color):
        if color is None:
            color = self.t['palette'][self._n % len(self.t['palette'])]; self._n += 1
        return color

    def add_vector(self, r, color=None, label=None, lw=3):
        """Arrow from the origin to Bloch vector r. Returns the artists (line, tip)."""
        color = self._color(color); r = np.asarray(r, float)
        line, = self.ax.plot([0, r[0]], [0, r[1]], [0, r[2]], color=color, lw=lw, label=label)
        tip, = self.ax.plot([r[0]], [r[1]], [r[2]], 'o', color=color, ms=7)
        return line, tip

    def add_state(self, state, **kw):
        """Arrow for a qubit state (vector, density matrix or Qiskit object)."""
        return self.add_vector(bloch_vectors(state, 1)[0], **kw)

    def add_points(self, points, color=None, size=12, alpha=0.9):
        p = np.atleast_2d(points)
        return self.ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=self._color(color), s=size, alpha=alpha)

    def add_trajectory(self, points, color=None, lw=1.8, alpha=0.8):
        p = np.atleast_2d(points)
        line, = self.ax.plot(p[:, 0], p[:, 1], p[:, 2], color=self._color(color), lw=lw, alpha=alpha)
        return line

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    def save(self, path, dpi=150):
        self.fig.savefig(path, dpi=dpi, facecolor=self.t['bg'], bbox_inches='tight')


def plot_bloch(vectors, names=None, theme='dark', title=None, size=3.6):
    """Static figure with one sphere per qubit. vectors: (qubits, 3) or a state (vector/density/Qiskit)."""
    import matplotlib.pyplot as plt
    v = np.asarray(vectors) if (np.ndim(vectors) == 2 and np.shape(vectors)[-1] == 3) else bloch_vectors(vectors)
    n = len(v); t = _theme(theme)
    fig = plt.figure(figsize=(size * n, size + 0.4), facecolor=t['bg'])
    spheres = []
    for q in range(n):
        b = BlochSphere(fig.add_subplot(1, n, q + 1, projection='3d'), (names or ['q%d' % i for i in range(n)])[q], theme)
        b.add_vector(v[q], t['palette'][q % len(t['palette'])]); spheres.append(b)
    if title:
        fig.suptitle(title, color=t['text'], fontsize=14)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.86 if title else 0.94, wspace=0)
    return fig, spheres


def animate_trajectory(traj, theme='dark', interval=40, trail=True, save=None, fps=25, size=3.6,
                       rotate=0.0, title=None, dpi=90):
    """Animate a Trajectory (one sphere per qubit). save: path ending in .gif (Pillow) or .mp4
    (ffmpeg). rotate: degrees of camera rotation per frame. Returns the FuncAnimation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    if not isinstance(traj, Trajectory):
        traj = Trajectory(traj)
    t = _theme(theme); n = traj.n_qubits; V = traj.vectors
    fig = plt.figure(figsize=(size * n, size + 0.6), facecolor=t['bg'])
    arts = []
    for q in range(n):
        b = BlochSphere(fig.add_subplot(1, n, q + 1, projection='3d'), traj.names[q], theme)
        c = t['palette'][q % len(t['palette'])]
        line, tip = b.add_vector(V[0, q], c)
        tr = b.add_trajectory(V[:1, q], c, lw=1.6, alpha=0.7) if trail else None
        arts.append((b, line, tip, tr))
    fig.subplots_adjust(left=0, right=1, bottom=0.03, top=0.84, wspace=0)
    caption = fig.text(0.5, 0.88, traj.labels[0], color=t['text'], ha='center', fontsize=11)
    if title or traj.title:
        fig.suptitle(title or traj.title, color=t['text'], fontsize=13, y=0.98)

    def update(k):
        out = []
        for q, (b, line, tip, tr) in enumerate(arts):
            r = V[k, q]
            line.set_data_3d([0, r[0]], [0, r[1]], [0, r[2]]); tip.set_data_3d([r[0]], [r[1]], [r[2]])
            out += [line, tip]
            if tr is not None:
                tr.set_data_3d(V[:k + 1, q, 0], V[:k + 1, q, 1], V[:k + 1, q, 2]); out.append(tr)
            if rotate:
                b.ax.view_init(elev=20, azim=35 + rotate * k)
        caption.set_text(traj.labels[k]); out.append(caption)
        return out

    anim = FuncAnimation(fig, update, frames=traj.frames, interval=interval, blit=False)
    if save:
        if str(save).endswith('.gif'):
            anim.save(save, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={'facecolor': t['bg']})
        else:
            anim.save(save, fps=fps, dpi=dpi, savefig_kwargs={'facecolor': t['bg']})
    return anim
