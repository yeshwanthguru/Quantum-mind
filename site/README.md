# Website

Source of https://yeshwanthguru.github.io/quantum-cognition-robotics/, built from this repository by
`.github/workflows/pages.yml` on every push to `main`.

| File | Role |
|---|---|
| `build.py` | builds `_site/`: landing page, interactive animations (from `qlcog.viz`), docs pages (from the Markdown files), executed notebooks |
| `templates_index.html`, `templates_page.html` | page templates |
| `static/style.css` | styles (dark and light, follows the system setting) |
| `static/playground.js` | the in-browser two-qubit playground (exact state-vector simulation) |
| `check_playground.js` | runs the playground's maths in Node.js; `tests/test_site.py` compares it with the package's simulator |

Local preview: `python3 site/build.py && python3 -m http.server -d _site 8000`, then open http://localhost:8000.
`--skip-notebooks` builds faster.
