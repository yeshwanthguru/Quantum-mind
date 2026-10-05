"""Bloch-sphere visualisation: watch qubits evolve in 3D.

  states        Bloch vectors, reduced states of entangled qubits, Trajectory, circuit_trajectory
                (gate-by-gate arcs from any Qiskit circuit), belief_trajectory (qlcog trust model),
                bloch_tomography (Bloch vectors measured on a simulator or quantum hardware)
  mpl           BlochSphere, plot_bloch, animate_trajectory (GIF / MP4)          needs matplotlib
  interactive   bloch_figure, animate_bloch (HTML with play/slider), LiveBloch   needs plotly
                (real-time updates in Jupyter or a Matplotlib window)

See README.md in this folder."""
from .states import (bloch_vector, state_from_bloch, reduced_density, bloch_vectors, Trajectory, circuit_trajectory,
                     belief_trajectory, rotation_trajectory, bloch_tomography, tomography_circuits)
from .themes import THEMES

__all__ = ['bloch_vector', 'state_from_bloch', 'reduced_density', 'bloch_vectors', 'Trajectory', 'circuit_trajectory',
           'belief_trajectory', 'rotation_trajectory', 'bloch_tomography', 'tomography_circuits', 'THEMES',
           'BlochSphere', 'plot_bloch', 'animate_trajectory', 'bloch_figure', 'animate_bloch', 'LiveBloch', 'save_html']


def __getattr__(name):            # plotting back ends are imported only when used
    if name in ('BlochSphere', 'plot_bloch', 'animate_trajectory'):
        from . import mpl
        return getattr(mpl, name)
    if name in ('bloch_figure', 'animate_bloch', 'LiveBloch', 'save_html'):
        from . import interactive
        return getattr(interactive, name)
    raise AttributeError(name)
