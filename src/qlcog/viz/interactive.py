"""Interactive Plotly Bloch spheres: rotate and zoom with the mouse, play animations with a slider,
export to a standalone HTML file, and update live in Jupyter (:class:`LiveBloch`).

Examples
--------
>>> from qiskit import QuantumCircuit                                  # doctest: +SKIP
>>> from qlcog.viz import circuit_trajectory, animate_bloch, save_html, LiveBloch
>>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)                       # doctest: +SKIP
>>> save_html(animate_bloch(circuit_trajectory(qc)), 'bell.html')      # doctest: +SKIP
>>> live = LiveBloch(2).show()                                         # doctest: +SKIP
>>> live.play(circuit_trajectory(qc), fps=30)                          # doctest: +SKIP
"""
from __future__ import annotations

import time
import numpy as np
from .themes import theme as _theme, AXIS_LABELS
from .states import Trajectory, bloch_vectors

__all__ = ['bloch_figure', 'animate_bloch', 'LiveBloch', 'save_html']

_DYN = 3          # dynamic traces per qubit: arrow, tip, trail


def _static_traces(t):
    """Sphere surface, wire-frame circles, axes and state labels (the parts that never move)."""
    from .._optional import require
    require('plotly')
    import plotly.graph_objects as go
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    tr = [go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), opacity=0.12,
                     colorscale=[[0, t['sphere']], [1, t['sphere']]], showscale=False, hoverinfo='skip')]
    th = np.linspace(0, 2 * np.pi, 120)
    circles = [(np.cos(th), np.sin(th), 0 * th), (np.cos(th), 0 * th, np.sin(th)), (0 * th, np.cos(th), np.sin(th))]
    for k in (-0.5, 0.5):
        r = np.sqrt(1 - k ** 2); circles.append((r * np.cos(th), r * np.sin(th), k + 0 * th))
    for x, y, z in circles:
        tr.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=t['wire'], width=2), hoverinfo='skip',
                               showlegend=False))
    for a in np.eye(3):
        tr.append(go.Scatter3d(x=[-a[0], a[0]], y=[-a[1], a[1]], z=[-a[2], a[2]], mode='lines',
                               line=dict(color=t['axis'], width=2, dash='dash'), hoverinfo='skip', showlegend=False))
    pos = np.array(list(AXIS_LABELS)); tr.append(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='text', text=list(AXIS_LABELS.values()),
        textfont=dict(color=t['text'], size=14), hoverinfo='skip', showlegend=False))
    return tr


def _dynamic_traces(r, trail, color, name):
    """Arrow, tip and trail of one qubit (the parts animations update)."""
    import plotly.graph_objects as go
    trail = np.atleast_2d(trail)
    return [go.Scatter3d(x=[0, r[0]], y=[0, r[1]], z=[0, r[2]], mode='lines', line=dict(color=color, width=9),
                         name=name, hoverinfo='skip'),
            go.Scatter3d(x=[r[0]], y=[r[1]], z=[r[2]], mode='markers', marker=dict(color=color, size=7),
                         hovertemplate=name + '<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
                         showlegend=False),
            go.Scatter3d(x=trail[:, 0], y=trail[:, 1], z=trail[:, 2], mode='lines',
                         line=dict(color=color, width=4), opacity=0.6, hoverinfo='skip', showlegend=False)]


def _layout(fig, n, t, title, height):
    """Shared 3D scene and page layout for n spheres."""
    scene = dict(xaxis=dict(visible=False, range=[-1.25, 1.25]), yaxis=dict(visible=False, range=[-1.25, 1.25]),
                 zaxis=dict(visible=False, range=[-1.25, 1.25]), aspectmode='cube',
                 camera=dict(eye=dict(x=1.35, y=1.0, z=0.7)), bgcolor=t['bg'])
    fig.update_layout(**{('scene' if q == 0 else 'scene%d' % (q + 1)): scene for q in range(n)})
    fig.update_layout(paper_bgcolor=t['bg'], plot_bgcolor=t['bg'], font=dict(color=t['text']),
                      title=dict(text=title or '', x=0.5), height=height, width=None,
                      margin=dict(l=10, r=10, t=60 if title else 30, b=10),
                      legend=dict(orientation='h', y=-0.02, x=0.5, xanchor='center'))


def _figure(vectors, trails, names, t, title, height, widget=False):
    """Build a figure with n spheres; returns it and the indices of the dynamic traces per qubit."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(vectors)
    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'scene'}] * n], subplot_titles=names,
                        horizontal_spacing=0.01)
    dyn_index = []
    for q in range(n):
        for tr in _static_traces(t):
            fig.add_trace(tr, row=1, col=q + 1)
    for q in range(n):
        c = t['palette'][q % len(t['palette'])]
        start = len(fig.data); dyn_index.append(list(range(start, start + _DYN)))
        for tr in _dynamic_traces(vectors[q], trails[q], c, names[q]):
            fig.add_trace(tr, row=1, col=q + 1)
    _layout(fig, n, t, title, height)
    if widget:
        from .._optional import require
        require('anywidget'); require('ipywidgets')          # Plotly >= 6 FigureWidget needs both
        fig = go.FigureWidget(fig)
    return fig, dyn_index


def bloch_figure(vectors, names=None, theme='dark', title=None, height=460):
    """Interactive figure with one sphere per qubit.

    Parameters
    ----------
    vectors : array_like or state
        Bloch vectors ``(qubits, 3)``, or a state.
    names : list of str, optional
    theme : str or dict, optional
    title : str, optional
    height : int, optional
        Height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    v = np.asarray(vectors, float) if (np.ndim(vectors) == 2 and np.shape(vectors)[-1] == 3) else bloch_vectors(vectors)
    names = names or ['q%d' % q for q in range(len(v))]
    fig, _ = _figure(v, [r[None] for r in v], names, _theme(theme), title, height)
    return fig


def animate_bloch(traj, theme='dark', fps=30, trail=True, title=None, height=500, max_frames=400):
    """Animated interactive figure with play and pause buttons and a slider labelled with the frame labels.

    Parameters
    ----------
    traj : Trajectory or array_like
    theme : str or dict, optional
    fps : int, optional
        Frames per second when playing.
    trail : bool, optional
        Draw the path followed so far.
    title : str, optional
    height : int, optional
    max_frames : int, optional
        Frames are subsampled to at most this many (keeps HTML files small).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    if not isinstance(traj, Trajectory):
        traj = Trajectory(traj)
    t = _theme(theme); V = traj.vectors
    keep = np.unique(np.linspace(0, traj.frames - 1, min(traj.frames, max_frames)).astype(int))
    fig, idx = _figure(V[0], [V[:1, q] for q in range(traj.n_qubits)], traj.names, t, title or traj.title, height)
    frames = []
    for k in keep:
        data = []
        for q in range(traj.n_qubits):
            data += _dynamic_traces(V[k, q], V[:k + 1, q] if trail else V[k:k + 1, q],
                                    t['palette'][q % len(t['palette'])], traj.names[q])
        frames.append(go.Frame(data=data, traces=sum(idx, []), name=str(k)))
    fig.frames = frames
    dur = 1000 / fps
    play = dict(label='▶ play', method='animate',
                args=[None, dict(frame=dict(duration=dur, redraw=True), transition=dict(duration=0), fromcurrent=True)])
    pause = dict(label='⏸ pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
    steps = [dict(method='animate', label=traj.labels[k], args=[[str(k)], dict(mode='immediate',
                  frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for k in keep]
    fig.update_layout(updatemenus=[dict(type='buttons', buttons=[play, pause], x=0.02, y=0.02, xanchor='left',
                                        yanchor='bottom', direction='left', bgcolor=t['wire'], font=dict(color=t['text']))],
                      sliders=[dict(steps=steps, x=0.2, len=0.78, y=0.02, currentvalue=dict(prefix='', font=dict(color=t['text'])),
                                    font=dict(color=t['bg'] if theme == 'light' else t['axis']))])
    return fig


def save_html(fig, path, auto_open=False):
    """Write a standalone HTML file (Plotly JavaScript loaded from its CDN).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    path : str
    auto_open : bool, optional
        Open the file in a browser.

    Returns
    -------
    str
        ``path``.
    """
    fig.write_html(path, include_plotlyjs='cdn', auto_open=auto_open)
    return path


class LiveBloch:
    """Real-time Bloch spheres.

    In Jupyter (with ipywidgets and anywidget) the figure is a Plotly ``FigureWidget`` updated in place;
    elsewhere a Matplotlib window is redrawn. Feed it states, Bloch vectors or whole trajectories.

    Parameters
    ----------
    n_qubits : int, optional
    names : list of str, optional
    theme : str or dict, optional
    backend : {'auto', 'plotly', 'matplotlib'}, optional
        ``'auto'`` picks Plotly inside a notebook and Matplotlib elsewhere.
    trail : bool, optional
        Keep and draw the history of every vector.
    title : str, optional

    Examples
    --------
    >>> live = LiveBloch(2).show()                    # doctest: +SKIP
    >>> for state in states:                          # doctest: +SKIP
    ...     live.update(state)
    >>> live.play(circuit_trajectory(qc), fps=30)     # doctest: +SKIP
    """

    def __init__(self, n_qubits=1, names=None, theme='dark', backend='auto', trail=True, title=None):
        self.n = n_qubits; self.names = names or ['q%d' % q for q in range(n_qubits)]
        self.t = _theme(theme); self.theme = theme; self.trail = trail; self.title = title
        if backend == 'auto':
            backend = 'plotly' if _in_notebook() else 'matplotlib'
        self.backend = backend; self.history = [[] for _ in range(n_qubits)]
        if backend == 'plotly':
            v0 = np.tile([0.0, 0.0, 1.0], (n_qubits, 1))
            self.fig, self.idx = _figure(v0, [v[None] for v in v0], self.names, self.t, title, 460, widget=True)
        else:
            import matplotlib.pyplot as plt
            from .mpl import BlochSphere
            plt.ion()
            self.fig = plt.figure(figsize=(3.6 * n_qubits, 4.0), facecolor=self.t['bg'])
            self.spheres = []
            for q in range(n_qubits):
                b = BlochSphere(self.fig.add_subplot(1, n_qubits, q + 1, projection='3d'), self.names[q], theme)
                c = self.t['palette'][q % len(self.t['palette'])]
                line, tip = b.add_vector([0, 0, 1], c); tr = b.add_trajectory([[0, 0, 1]], c)
                self.spheres.append((b, line, tip, tr))
            self.caption = self.fig.text(0.5, 0.03, '', color=self.t['text'], ha='center')

    def show(self):
        """Display the spheres (a widget in Jupyter, a window elsewhere).

        Returns
        -------
        LiveBloch
            ``self``.
        """
        if self.backend == 'plotly':
            from IPython.display import display
            display(self.fig)
        else:
            import matplotlib.pyplot as plt
            plt.show(block=False)
        return self

    def update(self, state, label=''):
        """Move the vectors to a new state.

        Parameters
        ----------
        state : array_like or Qiskit state
            Bloch vectors ``(n, 3)``, or a state vector, density matrix or Qiskit state.
        label : str, optional
            Caption.

        Returns
        -------
        LiveBloch
            ``self``.
        """
        v = np.asarray(state, float) if (np.ndim(state) == 2 and np.shape(state)[-1] == 3 and np.isrealobj(state)) \
            else bloch_vectors(state, self.n)
        for q in range(self.n):
            self.history[q].append(v[q])
        if self.backend == 'plotly':
            with self.fig.batch_update():
                for q in range(self.n):
                    a, tip, tr = (self.fig.data[i] for i in self.idx[q]); r = v[q]
                    a.x, a.y, a.z = [0, r[0]], [0, r[1]], [0, r[2]]
                    tip.x, tip.y, tip.z = [r[0]], [r[1]], [r[2]]
                    if self.trail:
                        h = np.array(self.history[q]); tr.x, tr.y, tr.z = h[:, 0], h[:, 1], h[:, 2]
                self.fig.layout.title.text = label or self.title or ''
        else:
            for q, (b, line, tip, tr) in enumerate(self.spheres):
                r = v[q]; line.set_data_3d([0, r[0]], [0, r[1]], [0, r[2]]); tip.set_data_3d([r[0]], [r[1]], [r[2]])
                if self.trail:
                    h = np.array(self.history[q]); tr.set_data_3d(h[:, 0], h[:, 1], h[:, 2])
            self.caption.set_text(label)
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        return self

    def play(self, traj, fps=30):
        """Replay a trajectory in real time.

        Parameters
        ----------
        traj : Trajectory or array_like
        fps : int, optional

        Returns
        -------
        LiveBloch
            ``self``.
        """
        if not isinstance(traj, Trajectory):
            traj = Trajectory(traj)
        for k in range(traj.frames):
            self.update(traj.vectors[k], traj.labels[k]); _sleep(1.0 / fps, self.backend)
        return self

    def reset(self):
        """Clear the trails.

        Returns
        -------
        LiveBloch
            ``self``.
        """
        self.history = [[] for _ in range(self.n)]; return self


def _sleep(dt, backend):
    """Wait between frames (Matplotlib needs plt.pause to redraw)."""
    if backend == 'matplotlib':
        import matplotlib.pyplot as plt
        plt.pause(dt)
    else:
        time.sleep(dt)


def _in_notebook():
    """True inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False
