"""A one-dimensional tabletop and a tiny instruction language for the one-motor VLA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLORS = ('red', 'green', 'blue')
WORDS = ('go', 'to', 'the', 'red', 'green', 'blue', 'leftmost', 'rightmost', 'block')
KINDS = ('color', 'leftmost', 'rightmost', 'block')


@dataclass
class Scene:
    """One episode: an image strip, an instruction, and the ground truth behind them.

    Attributes
    ----------
    image : (n_cells, 3) array
        Noisy one-hot colour strip; each object occupies one cell.
    text : (len(WORDS),) array
        Multi-hot bag of instruction words.
    cells, colors : lists
        Cell index and colour index of every object.
    valid : list of int
        Cells the instruction is true of. More than one means the instruction is ambiguous.
    target : int
        The cell the person actually means (drawn uniformly from ``valid``).
    """
    image: np.ndarray
    text: np.ndarray
    cells: list
    colors: list
    valid: list
    target: int
    words: tuple

    @property
    def ambiguous(self):
        return len(self.valid) > 1


def cell_centres(n_cells):
    return -1.0 + 2.0 * (np.arange(n_cells) + 0.5) / n_cells


def encode(words):
    bag = np.zeros(len(WORDS))
    for w in words:
        bag[WORDS.index(w)] = 1.0
    return bag


def _describe(cells, colors, target_cell):
    """A word that is true of the target cell and of nothing else, or None."""
    order = sorted(range(len(cells)), key=lambda i: cells[i])
    pos = [cells[i] for i in order]
    k = cells.index(target_cell)
    if target_cell == pos[0]:
        return 'leftmost'
    if target_cell == pos[-1]:
        return 'rightmost'
    if colors.count(colors[k]) == 1:
        return COLORS[colors[k]]
    return None


def _valid_cells(kind, word, cells, colors):
    order = sorted(cells)
    if kind == 'leftmost':
        return [order[0]]
    if kind == 'rightmost':
        return [order[-1]]
    if kind == 'block':
        return list(cells)
    return [c for c, col in zip(cells, colors) if COLORS[col] == word]


def sample_scene(rng, n_cells=16, pixel_noise=0.1, kinds=KINDS):
    """Draw a scene with 2-3 objects, an instruction, and the intended target.

    Scenes in which a middle object could not be named afterwards (same colour as another object, neither extreme)
    are redrawn, so a clarifying answer always exists.
    """
    while True:
        k = int(rng.integers(2, 4))
        cells = sorted(rng.choice(n_cells, k, replace=False).tolist())
        if min(np.diff(cells)) < 2:
            continue
        colors = rng.integers(0, 3, k).tolist()
        if all(_describe(cells, colors, c) is not None for c in cells):
            break
    kind = str(rng.choice(kinds))
    word = COLORS[colors[int(rng.integers(0, k))]] if kind == 'color' else kind
    valid = _valid_cells(kind, word, cells, colors)
    target = int(rng.choice(valid))
    words = ('go', 'to', 'the', word)
    img = pixel_noise * rng.standard_normal((n_cells, 3))
    for c, col in zip(cells, colors):
        img[c, col] += 1.0
    return Scene(img, encode(words), cells, colors, valid, target, words)


def clarification(scene, rng=None, n_cells=16, pixel_noise=0.1):
    """The person's answer to "which one?": a word that singles out the intended target."""
    word = _describe(scene.cells, scene.colors, scene.target)
    return encode(('go', 'to', 'the', word)), word
