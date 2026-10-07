"""Convert the tutorial sources (Markdown with ```python blocks) into Jupyter notebooks.

Text outside the code blocks becomes Markdown cells and every ```python block becomes a code cell, so
the tutorials stay readable on GitHub and run as notebooks (python tutorials/_convert.py out_dir).
"""
import pathlib
import re
import sys

import nbformat

HERE = pathlib.Path(__file__).resolve().parent


TITLE = re.compile(r"""(?:set_title|suptitle)\(\s*'([^']+)'|title='([^']+)'""")


def figure_alt(code, heading):
    """Alt text for the figures a code cell draws: its plot titles, or else the section heading."""
    titles = [a or b for a, b in TITLE.findall(code) if '%' not in (a or b)]
    return 'Figure: ' + ('; '.join(titles) if titles else heading)


def to_notebook(path):
    """Build a notebook from one tutorial source."""
    text = pathlib.Path(path).read_text()
    cells, heading = [], pathlib.Path(path).stem
    for i, part in enumerate(re.split(r'^```python\n(.*?)^```\n?', text, flags=re.S | re.M)):
        part = part.strip('\n')
        if not part:
            continue
        if i % 2:
            cell = nbformat.v4.new_code_cell(part)
            if 'plt.show()' in part:                                   # alt text for screen readers
                cell.metadata['mystnb'] = {'image': {'alt': figure_alt(part, heading)}}
            cells.append(cell)
        else:
            heads = re.findall(r'^#+ (.+)$', part, re.M)
            heading = heads[-1] if heads else heading
            cells.append(nbformat.v4.new_markdown_cell(part))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    return nb


def sources():
    """The tutorial sources in reading order."""
    return sorted(p for p in HERE.glob('[0-9][0-9]_*.md'))


if __name__ == '__main__':
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / 'notebooks')
    out.mkdir(parents=True, exist_ok=True)
    for src in sources():
        nbformat.write(to_notebook(src), out / (src.stem + '.ipynb'))
        print('wrote', out / (src.stem + '.ipynb'))
