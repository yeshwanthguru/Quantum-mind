"""Check every documentation reference against Crossref and build a BibTeX file (python docs/_ext/check_refs.py).

The references on the Foundations pages and in the model atlas were compiled by hand. This script

1. collects them (``docs/learn/*.md`` "References" sections and every atlas entry), resolves short forms
   such as "Howard (1966)" to the full reference cited elsewhere, and merges duplicates;
2. asks the Crossref API for the best match of each one and judges it: a match is *confirmed* when the
   first author's surname and the year agree;
3. writes ``references_report.md`` (every reference, its match, DOI and verdict, and where it is cited);
4. with ``--bib``, writes ``references.bib``: BibTeX from Crossref (through doi.org) for confirmed
   matches, and a clearly marked ``@misc`` entry with the text as cited for the others, so nothing is
   filled in from memory.

It needs internet access; run it from the "references" workflow in GitHub Actions and review the
report before committing ``references.bib``. :func:`collect` works offline and is tested.
"""
import argparse
import json
import pathlib
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
API = 'https://api.crossref.org/works?rows=1&query.bibliographic='
UA = {'User-Agent': 'quantum-mind-docs-check/1.1'}


def _plain(s):
    """Lower-case ASCII form of a string, for comparing names."""
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()


def _clean(ref):
    """Strip Markdown, 'related:' prefixes and trailing remarks in parentheses."""
    ref = ' '.join(ref.replace('*', '').split()).strip(' -')
    ref = re.sub(r'^related:\s*', '', ref)
    ref = re.sub(r'\s*\((adjoint differentiation|allocation QUBO)\)', '', ref)
    return ref.rstrip(' .') + '.'


def key_of(ref):
    """(first author's surname, year) of a reference, or None when no year is found."""
    m = re.match(r"\s*([^,(&]+?)(?:,| &| and |\s\()", ref.replace(' et al.', ''))
    y = re.search(r'\((\d{4})\)', ref)
    if not m or not y:
        return None
    return _plain(m.group(1).split()[-1]), y.group(1)


def _is_short(ref):
    """True for short citations such as 'Howard (1966).' or 'Busemeyer & Bruza (2024), ch. 4.'."""
    return bool(re.match(r"^[\w\-' ]+(?: & [\w\-' ]+)?(?: et al\.)? \(\d{4}\)(?:, ch\. \d+)?\.?$", ref))


def _raw():
    """(source, reference text) pairs from the Foundations pages and the model atlas."""
    out = []
    for page in sorted((ROOT / 'docs' / 'learn').glob('*.md')):
        text = page.read_text()
        if '## References' in text:
            block = text.split('## References', 1)[1]
            for item in re.split(r'\n- ', '\n' + block.strip()):
                if item.strip():
                    out.append((page.relative_to(ROOT).as_posix(), item))
    sys.path.insert(0, str(ROOT / 'docs' / '_ext'))
    import atlas
    for _, _, models in atlas.PAGES.values():
        for m in models:
            for part in re.split(r';\s*', m['ref'] or ''):
                part = re.sub(r'^trust from ', '', part.strip())
                for piece in re.split(r'\s+and\s+(?=[A-Z][\w\-]+ (?:et al\. )?\(\d{4}\))', part):
                    if piece.strip():
                        out.append(('atlas: ' + m['title'], piece))
    return out


def collect():
    """Unique references with every place each is cited.

    Returns
    -------
    refs : list of dict
        ``text`` (full reference), ``key`` (surname, year) and ``sources`` (list of str).
    unresolved : list of tuple
        ``(source, text)`` for short citations with no full reference anywhere.
    """
    raw = [(src, _clean(r)) for src, r in _raw()]
    full, by_key, unresolved = [], {}, []
    for src, r in raw:
        if _is_short(r):
            continue
        k = key_of(r)
        if k in by_key:                          # duplicate: keep the longest text
            e = by_key[k]
            e['sources'].append(src)
            if len(r) > len(e['text']):
                e['text'] = r
        else:
            e = {'text': r, 'key': k, 'sources': [src]}
            full.append(e)
            if k:
                by_key[k] = e
    for src, r in raw:
        if _is_short(r):
            k = key_of(r)
            if k in by_key:
                by_key[k]['sources'].append(src)
            else:
                unresolved.append((src, r))
    return full, unresolved


def lookup(ref):
    """Best Crossref match as a dict with ``title``, ``year``, ``doi`` and ``surnames``."""
    req = urllib.request.Request(API + urllib.parse.quote(ref), headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.load(r)['message']['items']
    if not items:
        return {}
    it = items[0]
    year = (it.get('issued', {}).get('date-parts') or [[None]])[0][0]
    return {'title': ' '.join((it.get('title') or [''])[0].split()), 'year': str(year or ''),
            'doi': it.get('DOI', ''), 'surnames': [_plain(a.get('family', '')) for a in it.get('author', [])]}


def our_title(ref):
    """The title in a reference ('Author (Year). Title. Journal ...'), or None when it gives only the journal."""
    m = re.search(r'\(\d{4}\)\.\s*(.+?)(?:\.\s|\.$|\s\()', ref)
    if not m or re.search(r',\s*\d', m.group(1)) or m.group(1).startswith('arXiv'):
        return None
    return m.group(1)


def _words(s):
    return set(re.findall(r'[a-z]{3,}', _plain(s)))


def verdict(entry, match):
    """'confirmed' when the first author and the year agree and, if our reference gives a title, Crossref's
    title matches it (80% of its words appear in our reference); otherwise 'check by hand'."""
    if not match or not entry['key']:
        return 'check by hand'
    surname, year = entry['key']
    if surname not in match['surnames'] or match['year'] != year:
        return 'check by hand'
    if our_title(entry['text']):
        theirs = _words(match.get('title', ''))
        if not theirs or len(theirs & _words(entry['text'])) < 0.8 * len(theirs):
            return 'check by hand'
    return 'confirmed'


def bibtex(doi):
    """BibTeX for a DOI from doi.org content negotiation."""
    req = urllib.request.Request('https://doi.org/' + doi, headers={**UA, 'Accept': 'application/x-bibtex'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8')


def _citekey(entry, used):
    surname, year = entry['key'] or ('ref', '')
    base = re.sub(r'\W', '', surname) + year
    key, n = base, 0
    while key in used:
        n += 1
        key = base + 'abcdefghij'[n - 1]
    used.add(key)
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bib', action='store_true', help='also write references.bib')
    args = ap.parse_args()
    refs, unresolved = collect()
    rows = ['# Reference check', '',
            'Best Crossref match for every reference in the Foundations pages and the model atlas. "confirmed" '
            'means the first author and the year agree; check every other row by hand (books often have no '
            'Crossref record).', '',
            '| Reference | Crossref title | Year | DOI | Verdict | Cited in |', '|---|---|---|---|---|---|']
    bib, used = ['% references.bib: generated by docs/_ext/check_refs.py --bib; review before committing.',
                 '% Entries from Crossref are marked "confirmed"; @misc entries hold the text as cited and need checking.', ''], set()
    for e in refs:
        try:
            match = lookup(e['text'])
        except Exception as exc:                  # network or API error: report it, keep going
            match = {'title': 'lookup failed: %s' % exc, 'year': '', 'doi': '', 'surnames': []}
        v = verdict(e, match)
        doi = match.get('doi', '')
        link = '[%s](https://doi.org/%s)' % (doi, doi) if doi else ''
        rows.append('| %s | %s | %s | %s | %s | %s |' % (e['text'].replace('|', '/'), match.get('title', '').replace('|', '/'),
                                                        match.get('year', ''), link, v, '; '.join(sorted(set(e['sources'])))))
        if args.bib:
            key = _citekey(e, used)
            bib.append('%% %s; cited in: %s' % (v, '; '.join(sorted(set(e['sources'])))))
            entry = None
            if v == 'confirmed' and doi:
                try:
                    entry = re.sub(r'^@(\w+)\{[^,]*,', r'@\1{%s,' % key, bibtex(doi).strip(), count=1)
                except Exception as exc:
                    bib.append('%% BibTeX download failed: %s' % exc)
            if entry is None:
                entry = '@misc{%s,\n  note = {UNVERIFIED, as cited: %s}\n}' % (key, e['text'].replace('{', '').replace('}', ''))
            bib += [entry, '']
        time.sleep(1)                             # be polite to the public APIs
    if unresolved:
        rows += ['', '## Short citations without a full reference', '']
        rows += ['- %s: %s' % (src, r) for src, r in unresolved]
    (ROOT / 'references_report.md').write_text('\n'.join(rows) + '\n')
    print('wrote', ROOT / 'references_report.md', '-', len(refs), 'references')
    if args.bib:
        (ROOT / 'references.bib').write_text('\n'.join(bib) + '\n')
        print('wrote', ROOT / 'references.bib')


if __name__ == '__main__':
    main()
