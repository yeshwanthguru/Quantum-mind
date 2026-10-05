"""Colour themes shared by the Matplotlib and Plotly Bloch spheres."""

THEMES = {
    'dark': dict(bg='#0d1117', sphere='#58a6ff', sphere_alpha=0.07, wire='#30363d', axis='#8b949e',
                 text='#e6edf3', palette=['#ff7b72', '#79c0ff', '#d2a8ff', '#7ee787', '#ffa657', '#f2cc60']),
    'light': dict(bg='#ffffff', sphere='#0969da', sphere_alpha=0.06, wire='#d0d7de', axis='#57606a',
                  text='#1f2328', palette=['#cf222e', '#0969da', '#8250df', '#1a7f37', '#bc4c00', '#9a6700']),
}

AXIS_LABELS = {(0, 0, 1.18): '|0⟩', (0, 0, -1.18): '|1⟩', (1.22, 0, 0): '|+⟩', (-1.22, 0, 0): '|−⟩',
               (0, 1.22, 0): '|+i⟩', (0, -1.22, 0): '|−i⟩'}


def theme(name):
    """Theme dict by name (dark, light) or a custom dict."""
    if isinstance(name, dict):
        return name
    return THEMES[name]
