"""Draw the logo and the example-gallery thumbnails (python3 docs/make_thumbs.py).

Each thumbnail is a small Bloch sphere in the colour of a pillar, so the gallery shows at a glance
which kind of model an example uses. The images are committed; rerun this script after changing it.
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / '_static'
BG = '#0d1117'
PILLARS = {                       # file name: (label, arrow colour, sphere colour)
    'ql': ('Quantum-like', '#ff7b72', '#58a6ff'),
    'q': ('Quantum', '#79c0ff', '#58a6ff'),
    'qi': ('Quantum-inspired', '#d2a8ff', '#a371f7'),
    'viz': ('Bloch viewer', '#7ee787', '#3fb950'),
    'robot': ('Robotics', '#ffa657', '#58a6ff'),
}


def sphere(ax, arrow, wire, theta=0.9, phi=0.7):
    """Draw a wire-frame Bloch sphere with one state vector on a 2D axis (oblique projection)."""
    t = np.linspace(0, 2 * np.pi, 200)
    ax.add_patch(plt.Circle((0, 0), 1, color=wire, alpha=0.10))
    ax.plot(np.cos(t), np.sin(t), color=wire, lw=2.2)
    ax.plot(np.cos(t), 0.28 * np.sin(t), color='#8b949e', lw=1.2, alpha=0.8)          # equator
    ax.plot(0.28 * np.sin(t), np.cos(t), color='#8b949e', lw=1.0, alpha=0.5)          # meridian
    x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    px, py = x - 0.28 * y, z + 0.28 * y * 0.3                                           # oblique view
    ax.annotate('', xy=(px, py), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=arrow, lw=4, mutation_scale=22))
    ax.plot([0], [0], 'o', color='#e6edf3', ms=4)


def thumbnail(name, label, arrow, wire):
    """Save one 400 x 280 thumbnail."""
    fig = plt.figure(figsize=(4, 2.8), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.0, 0.26, 1.0, 0.72])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect('equal')
    ax.axis('off')
    sphere(ax, arrow, wire)
    fig.text(0.5, 0.06, label, color=arrow, ha='center', fontsize=24, fontweight='bold')
    fig.savefig(OUT / 'thumbs' / ('%s.png' % name), facecolor=BG)
    plt.close(fig)


def logo():
    """Square logo used in the documentation header."""
    fig = plt.figure(figsize=(2, 2), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    sphere(ax, '#ff7b72', '#58a6ff')
    fig.savefig(OUT / 'logo.png', transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    (OUT / 'thumbs').mkdir(parents=True, exist_ok=True)
    for key, (label, arrow, wire) in PILLARS.items():
        thumbnail(key, label, arrow, wire)
    logo()
    print('thumbnails and logo written to', OUT)
