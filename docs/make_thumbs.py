"""Draw the example-gallery thumbnails (python3 docs/make_thumbs.py).

Each example gets its own thumbnail: a small Bloch sphere in the colour of its pillar, its number and
its title, so the gallery shows at a glance what each example is about and which kind of model it
uses. The pillar comes from the ``sphinx_gallery_thumbnail_path`` line of the example, which this
script also points at the example's own thumbnail. The images are committed; rerun this script after
adding or renaming an example.
"""
import pathlib
import re
import textwrap

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


def example_thumbnail(number, title, pillar):
    """Save the 400 x 280 thumbnail of one example."""
    label, arrow, wire = PILLARS[pillar]
    fig = plt.figure(figsize=(4, 2.8), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.02, 0.30, 0.42, 0.62])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    sphere(ax, arrow, wire, theta=0.4 + 0.07 * number, phi=0.3 * number)
    fig.text(0.95, 0.62, '%02d' % number, color=arrow, ha='right', va='center', fontsize=54, fontweight='bold')
    fig.text(0.95, 0.36, label, color='#8b949e', ha='right', va='center', fontsize=12)
    fig.text(0.05, 0.10, '\n'.join(textwrap.wrap(title, 38)[:2]), color='#e6edf3', ha='left', va='bottom',
             fontsize=12.5, fontweight='bold')
    fig.savefig(OUT / 'thumbs' / ('ex_%02d.png' % number), facecolor=BG)
    plt.close(fig)


def examples():
    """Draw a thumbnail per example and point each example at its own thumbnail."""
    root = pathlib.Path(__file__).resolve().parents[1] / 'examples'
    pattern = re.compile(r"# sphinx_gallery_thumbnail_path = '_static/thumbs/(\w+)\.png'")
    for f in sorted(root.glob('[0-9][0-9]_*.py')):
        src = f.read_text()
        number, title = int(f.name[:2]), src.split('\n', 1)[0].strip('"\' ')
        m = pattern.search(src)
        key = m.group(1) if m else 'q'
        pillar = key if key in PILLARS else META.get(f.name, 'q')
        pattern_own = "# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_%02d.png'" % number
        if not m and pattern_own not in src:
            raise SystemExit('%s has no sphinx_gallery_thumbnail_path line' % f.name)
        example_thumbnail(number, title, pillar)
        line = "# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_%02d.png'" % number
        new = pattern.sub(line, src) if m else src
        if new != src:
            f.write_text(new)
        META[f.name] = pillar


#: Pillar of each example once its thumbnail line points at its own image (kept in thumbs/pillars.txt).
META = {}

if __name__ == '__main__':
    (OUT / 'thumbs').mkdir(parents=True, exist_ok=True)
    for key, (label, arrow, wire) in PILLARS.items():
        thumbnail(key, label, arrow, wire)
    store = OUT / 'thumbs' / 'pillars.txt'
    if store.exists():
        META.update(dict(line.split() for line in store.read_text().splitlines() if line.strip()))
    examples()
    store.write_text(''.join('%s %s\n' % kv for kv in sorted(META.items())))
    print('thumbnails written to', OUT / 'thumbs')
