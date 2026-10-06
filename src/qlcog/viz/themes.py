"""Colour themes shared by the Matplotlib and Plotly Bloch spheres."""

#: Built-in colour themes: background, sphere, wire-frame, axis and text colours, and a palette for vectors.
THEMES = {
    'dark': dict(bg='#0d1117', sphere='#58a6ff', sphere_alpha=0.07, wire='#30363d', axis='#8b949e',
                 text='#e6edf3', palette=['#ff7b72', '#79c0ff', '#d2a8ff', '#7ee787', '#ffa657', '#f2cc60']),
    'light': dict(bg='#ffffff', sphere='#0969da', sphere_alpha=0.06, wire='#d0d7de', axis='#57606a',
                  text='#1f2328', palette=['#cf222e', '#0969da', '#8250df', '#1a7f37', '#bc4c00', '#9a6700']),
}

#: Positions and labels of the six cardinal states.
AXIS_LABELS = {(0, 0, 1.18): '|0⟩', (0, 0, -1.18): '|1⟩', (1.22, 0, 0): '|+⟩', (-1.22, 0, 0): '|−⟩',
               (0, 1.22, 0): '|+i⟩', (0, -1.22, 0): '|−i⟩'}


def theme(name):
    """Look up a colour theme.

    Parameters
    ----------
    name : str or dict
        ``'dark'`` or ``'light'``, or a custom dict with the keys of ``THEMES['dark']``.

    Returns
    -------
    dict
    """
    if isinstance(name, dict):
        return name
    return THEMES[name]
