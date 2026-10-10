"""Group the reference shells by embedding (k-means on tight-crop embeddings).

Writes analysis/out/shell_groups.html: each group's most typical shells, with
its views and most common turtles. Groups are only loosely separated
(silhouette ~0.17); roughly half reflect shell pattern (dense spots, bold
contrast, radiating streaks, sparse brown) and half photo conditions (dirt,
faded light, plants in frame).

    K=8 python analysis/shell_groups.py
"""
import collections, csv, html, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, os.getcwd())
import identify, matching, shellcrop

ROOT = os.getcwd(); K = int(os.environ.get("K", 8))
OUT = os.path.join(ROOT, "analysis", "out"); os.makedirs(OUT, exist_ok=True)
names, paths, X = identify.load_reference(identify.DROP, identify.DATA / shellcrop.VARIANTS["tight"]["db"])
names = np.array(names); ids = [matching.capture_id_of(p) for p in paths]
meta = {int(r["capture_id"]): r for r in csv.DictReader(open("data/capture_meta.csv"))}
views = np.array([meta[c]["view"].replace("carapace_", "") for c in ids])
P = matching.unit(PCA(32, random_state=0).fit_transform(X))
km = KMeans(K, n_init=10, random_state=0).fit(P); lab = km.labels_

cons = [collections.Counter(lab[names == n]).most_common(1)[0][1] / (names == n).sum() for n in sorted(set(names))]
print(f"average share of a turtle's photos in its most common group: {np.mean(cons):.0%} (if groups ignored the turtle: ~{1/K:.0%}-25%)")

secs = []
for rank, c in enumerate(np.argsort(-np.bincount(lab)), 1):
    m = np.where(lab == c)[0]
    centre = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
    m = m[np.argsort(-(P[m] @ centre))]
    vc = collections.Counter(views[m]).most_common(3)
    tc = collections.Counter(names[m]).most_common(5)
    vtxt = ", ".join(f"{v} {n}" for v, n in vc)
    ttxt = ", ".join(f"{html.escape(t)} {n}" for t, n in tc)
    print(f"group {rank}: {len(m)} photos; views {vtxt}; turtles {ttxt}")
    imgs = "".join(
        f'<figure><img loading="lazy" src="file://{ROOT}/data/crops_tight/{names[j]}/{ids[j]}.jpg">'
        f'<figcaption>{html.escape(names[j])} · {views[j]}</figcaption></figure>' for j in m[:40])
    secs.append(f'<section><h2>Group {rank} · {len(m)} photos</h2><p class="m">Views: {vtxt} · '
                f'Most common turtles: {ttxt} · showing the 40 most typical</p><div class="g">{imgs}</div></section>')

page = ('<!doctype html><meta charset="utf-8"><title>Shell Groups</title><style>'
        'body{font:14px system-ui;margin:16px;background:#f6f4ee;color:#1d2a22}h2{font-size:17px;margin:20px 0 2px}'
        '.m{color:#6b756e;margin:0 0 8px}.g{display:flex;flex-wrap:wrap;gap:6px}figure{margin:0}'
        'img{height:96px;border-radius:6px;display:block}figcaption{font-size:11px;color:#6b756e}</style>'
        f'<h1>Reference shells grouped by embedding (tight crops, {K} groups)</h1>'
        '<p class="m">Automatic k-means groups of the 628 reference photos; each group shows its most typical members first.</p>'
        + "".join(secs))
open(f"{OUT}/shell_groups.html", "w").write(page)
