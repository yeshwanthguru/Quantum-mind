"""Look every documentation reference up in Crossref and write a report (python docs/_ext/check_refs.py).

The references on the Foundations pages were compiled by hand. This script extracts each one, asks the
Crossref API for the best bibliographic match, and writes ``references_report.md`` with the matched
title, year, DOI and a similarity score, so that each reference can be confirmed or corrected and the
DOIs added. It needs internet access; run it from the "references" workflow in GitHub Actions.
"""
import difflib
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
API = 'https://api.crossref.org/works?rows=1&query.bibliographic='


def references():
    """(page, reference) pairs from the 'References' sections of the documentation pages."""
    out = []
    for page in sorted((ROOT / 'docs' / 'learn').glob('*.md')):
        text = page.read_text()
        if '## References' not in text:
            continue
        block = text.split('## References', 1)[1]
        for item in re.split(r'\n- ', '\n' + block.strip()):
            item = ' '.join(item.split()).strip('- ')
            if item:
                out.append((page.relative_to(ROOT).as_posix(), item))
    return out


def lookup(ref):
    """Best Crossref match: (title, year, doi, similarity of our reference to the match)."""
    req = urllib.request.Request(API + urllib.parse.quote(ref), headers={'User-Agent': 'quantum-mind-docs-check/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.load(r)['message']['items']
    if not items:
        return '', '', '', 0.0
    it = items[0]
    title = ' '.join(it.get('title', [''])[0].split())
    year = str((it.get('issued', {}).get('date-parts') or [[None]])[0][0] or '')
    score = difflib.SequenceMatcher(None, title.lower(), ref.lower()).find_longest_match(0, len(title), 0, len(ref)).size
    return title, year, it.get('DOI', ''), score / max(len(title), 1)


def main():
    rows = ['# Reference check', '', 'Best Crossref match for every reference on the Foundations pages. A score near 1 '
            'means the matched title appears in our reference; check every row below 0.8 by hand. Books often have no '
            'Crossref record, so a low score for a book is expected.', '',
            '| Page | Our reference | Crossref title | Year | DOI | Score |', '|---|---|---|---|---|---|']
    for page, ref in references():
        try:
            title, year, doi, score = lookup(ref)
        except Exception as exc:                      # network or API error: report it, keep going
            title, year, doi, score = 'lookup failed: %s' % exc, '', '', 0.0
        link = '[%s](https://doi.org/%s)' % (doi, doi) if doi else ''
        rows.append('| %s | %s | %s | %s | %s | %.2f |' % (page, ref.replace('|', '/'), title.replace('|', '/'), year, link, score))
        time.sleep(1)                                 # be polite to the public API
    (ROOT / 'references_report.md').write_text('\n'.join(rows) + '\n')
    print('wrote', ROOT / 'references_report.md', '-', len(rows) - 6, 'references')


if __name__ == '__main__':
    main()
