# Website

Source of https://yeshwanthguru.github.io/quantum-cognition-robotics/, built from this repository by
`.github/workflows/pages.yml` on every push to `main`.

| File | Role |
|---|---|
| `build.py` | builds `_site/`: landing page, interactive animations (from `qlcog.viz`), and the Sphinx documentation from `docs/` into `_site/docs/` |
| `templates_index.html`, `templates_page.html` | landing page and 404 page templates |
| `static/style.css` | styles (dark and light, follows the system setting) |
| `static/playground.js` | the in-browser two-qubit playground (exact state-vector simulation) |
| `check_playground.js` | runs the playground's maths in Node.js; `tests/test_site.py` compares it with the package's simulator |

Local preview: `python3 site/build.py && python3 -m http.server -d _site 8000`, then open http://localhost:8000.
`--fast` skips running the examples and notebooks in the documentation; `--skip-docs` builds the
landing page only. The documentation alone: `python -m sphinx -b html docs docs/_build/html`.
