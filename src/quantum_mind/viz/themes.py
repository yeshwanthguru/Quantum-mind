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


def use_mpl_style(name='dark'):
    """Apply a theme to every Matplotlib figure that follows (used by the tutorials).

    Parameters
    ----------
    name : str or dict, optional
        ``'dark'`` or ``'light'``, or a custom dict with the keys of ``THEMES['dark']``.

    Returns
    -------
    dict
        The theme that was applied.
    """
    import matplotlib as mpl
    from cycler import cycler
    t = theme(name)
    mpl.rcParams.update({
        'figure.facecolor': t['bg'], 'axes.facecolor': t['bg'], 'savefig.facecolor': t['bg'],
        'axes.edgecolor': t['wire'], 'axes.labelcolor': t['text'], 'text.color': t['text'],
        'xtick.color': t['axis'], 'ytick.color': t['axis'], 'grid.color': t['wire'], 'axes.grid': True,
        'grid.alpha': 0.6, 'axes.prop_cycle': cycler(color=t['palette']), 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.dpi': 110, 'font.size': 10,
        'legend.frameon': False, 'image.cmap': 'magma'})
    return t
