"""Matplotlib Bloch spheres: static figures, animations, GIF and MP4 export, an entanglement timeline and
the Q-sphere.

Examples
--------
>>> from qiskit import QuantumCircuit                       # doctest: +SKIP
>>> from qlcog.viz import circuit_trajectory, animate_trajectory, plot_entanglement
>>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)            # doctest: +SKIP
>>> traj = circuit_trajectory(qc, steps=15)                 # doctest: +SKIP
>>> animate_trajectory(traj, save='bell.gif', rotate=0.5)   # doctest: +SKIP
>>> plot_entanglement(traj).savefig('bell_timeline.png')    # doctest: +SKIP
"""
from __future__ import annotations

import numpy as np
from .themes import theme as _theme, AXIS_LABELS
from .states import Trajectory, bloch_vectors

__all__ = ['BlochSphere', 'plot_bloch', 'animate_trajectory', 'plot_entanglement', 'plot_qsphere']


class BlochSphere:
    """One Bloch sphere on a Matplotlib 3D axis.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Axis to draw on (a new figure by default).
    title : str, optional
        Caption under the sphere.
    theme : str or dict, optional
        ``'dark'`` (default), ``'light'`` or a custom theme.
    figsize : tuple, optional
        Figure size for a new figure.
    elev, azim : float, optional
        Camera angles in degrees.

    Examples
    --------
    >>> b = BlochSphere(title='q0')            # doctest: +SKIP
    >>> b.add_vector([0, 1, 0])                # doctest: +SKIP
    >>> b.save('sphere.png')                   # doctest: +SKIP
    """

    def __init__(self, ax=None, title='', theme='dark', figsize=(5, 5), elev=20, azim=35):
        from .._optional import require
        require('matplotlib')
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
        """Draw the translucent sphere, latitude and meridian circles, axes and state labels."""
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
        """Use the given colour, or the next colour of the theme palette."""
        if color is None:
            color = self.t['palette'][self._n % len(self.t['palette'])]; self._n += 1
        return color

    def add_vector(self, r, color=None, label=None, lw=3):
        """Arrow from the origin to a Bloch vector.

        Parameters
        ----------
        r : array_like
            ``(x, y, z)``.
        color : str, optional
            Colour (next palette colour by default).
        label : str, optional
            Legend label.
        lw : float, optional
            Line width.

        Returns
        -------
        tuple
            ``(line, tip)`` artists, which animations update.
        """
        color = self._color(color); r = np.asarray(r, float)
        line, = self.ax.plot([0, r[0]], [0, r[1]], [0, r[2]], color=color, lw=lw, label=label)
        tip, = self.ax.plot([r[0]], [r[1]], [r[2]], 'o', color=color, ms=7)
        return line, tip

    def add_state(self, state, **kw):
        """Arrow for a qubit state.

        Parameters
        ----------
        state : array_like or Qiskit state
            2-vector, 2 x 2 density matrix, ``Statevector`` or ``DensityMatrix``.
        **kw
            Passed to :meth:`add_vector`.

        Returns
        -------
        tuple
        """
        return self.add_vector(bloch_vectors(state, 1)[0], **kw)

    def add_points(self, points, color=None, size=12, alpha=0.9):
        """Scatter points on or inside the sphere.

        Parameters
        ----------
        points : array_like
            Shape ``(n, 3)``.
        color : str, optional
        size : float, optional
        alpha : float, optional

        Returns
        -------
        matplotlib artist
        """
        p = np.atleast_2d(points)
        return self.ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=self._color(color), s=size, alpha=alpha)

    def add_trajectory(self, points, color=None, lw=1.8, alpha=0.8):
        """Draw a path of Bloch vectors.

        Parameters
        ----------
        points : array_like
            Shape ``(n, 3)``.
        color : str, optional
        lw : float, optional
        alpha : float, optional

        Returns
        -------
        matplotlib line
        """
        p = np.atleast_2d(points)
        line, = self.ax.plot(p[:, 0], p[:, 1], p[:, 2], color=self._color(color), lw=lw, alpha=alpha)
        return line

    def show(self):
        """Show the figure."""
        import matplotlib.pyplot as plt
        plt.show()

    def save(self, path, dpi=150):
        """Save the figure.

        Parameters
        ----------
        path : str
            File name (format from the extension).
        dpi : int, optional
        """
        self.fig.savefig(path, dpi=dpi, facecolor=self.t['bg'], bbox_inches='tight')


def plot_bloch(vectors, names=None, theme='dark', title=None, size=3.6):
    """Static figure with one sphere per qubit.

    Parameters
    ----------
    vectors : array_like or state
        Bloch vectors ``(qubits, 3)``, or a state vector, density matrix or Qiskit state.
    names : list of str, optional
        Qubit names.
    theme : str or dict, optional
    title : str, optional
    size : float, optional
        Size per sphere in inches.

    Returns
    -------
    figure : matplotlib.figure.Figure
    spheres : list of BlochSphere
    """
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
    """Animate a trajectory, one sphere per qubit.

    Parameters
    ----------
    traj : Trajectory or array_like
        Trajectory, or Bloch vectors ``(frames, qubits, 3)``.
    theme : str or dict, optional
    interval : int, optional
        Milliseconds between frames on screen.
    trail : bool, optional
        Draw the path followed so far.
    save : str, optional
        Path ending in ``.gif`` (Pillow) or ``.mp4`` (ffmpeg).
    fps : int, optional
        Frames per second of the saved file.
    size : float, optional
        Size per sphere in inches.
    rotate : float, optional
        Camera rotation per frame, in degrees.
    title : str, optional
    dpi : int, optional
        Resolution of the saved file.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
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


def plot_entanglement(traj, pairs=None, theme='dark', size=(8, 3.2)):
    """Entanglement timeline of a circuit trajectory.

    Plots the Bloch-vector length of every qubit (1 = not entangled with the others, for a pure global
    state) and the concurrence of qubit pairs (0 = separable, 1 = maximally entangled), with the gate
    labels on the x axis.

    Parameters
    ----------
    traj : Trajectory
        Circuit trajectory (with stored states, for the concurrence).
    pairs : list of tuple, optional
        Qubit pairs (default: up to six pairs).
    theme : str or dict, optional
    size : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .._optional import require
    require('matplotlib')
    import matplotlib.pyplot as plt
    t = _theme(theme); n = traj.n_qubits
    pairs = pairs if pairs is not None else [(a, b) for a in range(n) for b in range(a + 1, n)][:6]
    fig, ax = plt.subplots(figsize=size, facecolor=t['bg']); ax.set_facecolor(t['bg'])
    x = np.arange(traj.frames); L = traj.purity()
    for q in range(n):                     # decreasing widths keep coinciding lines visible
        ax.plot(x, L[:, q], color=t['palette'][q % len(t['palette'])], lw=2 + 2.5 * (n - 1 - q) / max(n - 1, 1),
                label='|r| ' + traj.names[q], alpha=0.9)
    if traj.states:
        for k, (a, b) in enumerate(pairs):
            ax.plot(x, traj.concurrence(a, b), color=t['text'], lw=1.6, ls=['--', ':', '-.'][k % 3],
                    label='concurrence %s-%s' % (traj.names[a], traj.names[b]))
    starts = [k for k in range(1, traj.frames) if traj.labels[k] != traj.labels[k - 1]] + [traj.frames]
    centres = [(a + b - 1) / 2 for a, b in zip(starts[:-1], starts[1:])]     # one label per gate
    ax.set_xticks(centres); ax.set_xticklabels([traj.labels[int(c)] for c in centres], fontsize=8)
    for a in starts[:-1]:
        ax.axvline(a - 0.5, color=t['wire'], lw=0.6)
    ax.set_ylim(-0.03, 1.05); ax.set_ylabel('value', color=t['text']); ax.tick_params(colors=t['text'])
    for sp in ax.spines.values():
        sp.set_color(t['wire'])
    ax.grid(axis='y', color=t['wire'], lw=0.5)
    ax.legend(facecolor=t['bg'], edgecolor=t['wire'], labelcolor=t['text'], fontsize=8, loc='lower left')
    fig.tight_layout()
    return fig


def plot_qsphere(state, **kw):
    """Q-sphere of a multi-qubit state, drawn by Qiskit's ``plot_state_qsphere``.

    Basis states are placed by Hamming weight, with amplitude as size and phase as colour.

    Parameters
    ----------
    state : array_like or qiskit.quantum_info.Statevector
    **kw
        Passed to ``plot_state_qsphere``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .._optional import require
    require('qiskit'); require('matplotlib'); require('seaborn', 'viz')     # Qiskit's Q-sphere needs seaborn
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_state_qsphere
    return plot_state_qsphere(state if hasattr(state, 'data') else Statevector(np.asarray(state, complex)), **kw)
