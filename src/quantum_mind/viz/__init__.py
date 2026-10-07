"""Bloch-sphere visualisation: watch qubits evolve in 3D.

* :mod:`~quantum_mind.viz.states`: Bloch vectors, reduced states of entangled qubits, concurrence,
  :class:`Trajectory`, :func:`circuit_trajectory` (gate-by-gate arcs from any Qiskit circuit),
  :func:`belief_trajectory` (the trust model) and :func:`bloch_tomography` (Bloch vectors measured on a
  simulator or quantum hardware). NumPy only.
* :mod:`~quantum_mind.viz.mpl`: ``BlochSphere``, ``plot_bloch``, ``animate_trajectory`` (GIF and MP4),
  ``plot_entanglement`` and ``plot_qsphere``. Needs Matplotlib.
* :mod:`~quantum_mind.viz.interactive`: ``bloch_figure``, ``animate_bloch`` (HTML with play and slider) and
  ``LiveBloch`` (real-time updates in Jupyter or a Matplotlib window). Needs Plotly.

The plotting back ends are imported only when first used. See README.md in this folder.
"""
from .states import (bloch_vector, state_from_bloch, reduced_density, reduced_density_pair, concurrence,
                     entanglement_summary, bloch_vectors, Trajectory, circuit_trajectory, belief_trajectory,
                     rotation_trajectory, bloch_tomography, tomography_circuits)
from .themes import THEMES, use_mpl_style

__all__ = ['bloch_vector', 'state_from_bloch', 'reduced_density', 'reduced_density_pair', 'concurrence',
           'entanglement_summary', 'plot_entanglement', 'plot_qsphere', 'bloch_vectors', 'Trajectory', 'circuit_trajectory',
           'belief_trajectory', 'rotation_trajectory', 'bloch_tomography', 'tomography_circuits', 'THEMES',
           'BlochSphere', 'plot_bloch', 'animate_trajectory', 'bloch_figure', 'animate_bloch', 'LiveBloch', 'save_html',
           'use_mpl_style']


def __getattr__(name):            # plotting back ends are imported only when used
    if name in ('BlochSphere', 'plot_bloch', 'animate_trajectory', 'plot_entanglement', 'plot_qsphere'):
        from . import mpl
        return getattr(mpl, name)
    if name in ('bloch_figure', 'animate_bloch', 'LiveBloch', 'save_html'):
        from . import interactive
        return getattr(interactive, name)
    raise AttributeError(name)
