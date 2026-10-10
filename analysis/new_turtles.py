"""Group unmatched field photos into candidate new turtles.

Photos whose best spot score on file is below MAX_SPOT (default 1) are
SIFT-matched against each other; pairs scoring >= LINK (default 4) are
linked, and connected photos form one candidate new turtle. Linking is
transitive, so one weak link can chain two turtles: check groups by eye.
Needs the session's shell crops in uploads/. Writes analysis/out/new_turtles.html.

    python analysis/new_turtles.py [results-folder]
    MAX_SPOT=1.5 LINK=6 python analysis/new_turtles.py
"""
import csv, html, itertools, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.getcwd())
import sift_match

ROOT = os.getcwd(); DATA = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = os.path.join(ROOT, "analysis", "out"); os.makedirs(OUT, exist_ok=True)
MAX_SPOT = float(os.environ.get("MAX_SPOT", 1.0))   # photos whose best match on file scored below this
LINK = float(os.environ.get("LINK", 4.0))           # spot score that links two photos as the same turtle

rows = list(csv.DictReader(open(f"{DATA}/uploads_summary.csv")))
meta = {r["upload_id"]: r["taken"] for r in csv.DictReader(open(f"{DATA}/photo_meta.csv"))}
pick = [r for r in rows if r["best_spot_score"] and float(r["best_spot_score"]) < MAX_SPOT
        and os.path.exists(f"uploads/{r['upload_id']}_crop.jpg")]
print(f"{len(pick)} photos with best spot score < {MAX_SPOT} on file")

feats = list(ThreadPoolExecutor(8).map(lambda r: sift_match.features(f"uploads/{r['upload_id']}_crop.jpg"), pick))
pairs = list(itertools.combinations(range(len(pick)), 2))
scores = list(ThreadPoolExecutor(8).map(lambda ij: sift_match.score(feats[ij[0]], feats[ij[1]]), pairs))
M = np.zeros((len(pick), len(pick)))
for (i, j), s in zip(pairs, scores):
    M[i, j] = M[j, i] = s

# connected components over strong links
parent = list(range(len(pick)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for (i, j), s in zip(pairs, scores):
    if s >= LINK:
        parent[find(i)] = find(j)
groups = {}
for i in range(len(pick)):
    groups.setdefault(find(i), []).append(i)
multi = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
singles = [g[0] for g in groups.values() if len(g) == 1]
day = lambda i: (meta.get(pick[i]["upload_id"]) or pick[i]["time"])[:10]
cross = sum(len({day(i) for i in g}) > 1 for g in multi)
print(f"{len(multi)} candidate turtles (2+ photos linked at spot >= {LINK}), {cross} of them seen on 2+ days; "
      f"{len(singles)} photos match nothing")
print("sizes:", [len(g) for g in multi])

def tile(i, extra=""):
    r = pick[i]; t = meta.get(r["upload_id"]) or r["time"].replace("T", " ")
    return (f'<figure><a href="file://{ROOT}/uploads/{r["upload_id"]}.jpg" target="_blank">'
            f'<img loading="lazy" src="file://{ROOT}/uploads/{r["upload_id"]}_crop.jpg"></a>'
            f'<figcaption>{html.escape(r["filename"])}<br>{html.escape(t[:16])}{extra}</figcaption></figure>')

secs = []
for k, g in enumerate(multi, 1):
    g = sorted(g, key=lambda i: meta.get(pick[i]["upload_id"]) or pick[i]["time"])
    sub = M[np.ix_(g, g)]; strong = sub[np.triu_indices(len(g), 1)]
    days = sorted({day(i) for i in g})
    guess = {pick[i]["best_pick"] for i in g}
    secs.append(f'<section><h2>Candidate new turtle {k} · {len(g)} photos · {len(days)} day{"s" if len(days) > 1 else ""}</h2>'
                f'<p class="m">Days: {", ".join(days)} · strongest link between its photos: spot {strong.max():.1f} · '
                f'best guesses on file (all weak): {html.escape(", ".join(sorted(guess)))}</p>'
                f'<div class="g">{"".join(tile(i) for i in g)}</div></section>')
singles_html = "".join(tile(i) for i in sorted(singles, key=lambda i: meta.get(pick[i]["upload_id"]) or ""))
page = ('<!doctype html><meta charset="utf-8"><title>Candidate New Turtles</title><style>'
        'body{font:14px system-ui;margin:16px;background:#f6f4ee;color:#1d2a22}h2{font-size:17px;margin:20px 0 2px}'
        '.m{color:#6b756e;margin:0 0 8px}.g{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0}'
        'img{height:110px;border-radius:6px;display:block}figcaption{font-size:11px;color:#6b756e}</style>'
        f'<h1>Candidate new turtles</h1><p class="m">{len(pick)} field photos had no spot match on file (best score &lt; {MAX_SPOT:g}). '
        f'Photos whose shells spot-match each other (score ≥ {LINK:g}) are grouped as one candidate turtle. '
        f'{len(multi)} groups, {cross} seen on more than one day; {len(singles)} photos match nothing. Click a photo for the full image.</p>'
        + "".join(secs) + f'<h2>Unmatched singles · {len(singles)}</h2><div class="g">{singles_html}</div>')
open(f"{OUT}/new_turtles.html", "w").write(page)
