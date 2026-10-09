"""Shared matching maths for app.py and evaluate.py.

A matcher turns reference embeddings into a space where cosine similarity
means "same turtle":

  PCA  — unsupervised: subtract the mean photo, keep the top `dim` directions
  LDA  — PCA first, then Linear Discriminant Analysis using turtle names:
         directions where different turtles differ and the same turtle stays
         put across photos (shrinkage="auto" because most turtles have few
         photos)

Honest similarity ("out-of-fold"): LDA learns from names, so scoring a
reference photo against the others with a projection that saw it is
optimistic. oof_sims() fits on 4/5 of the (turtle, day) groups and fills in
the held-out rows, the same way a new sighting would be scored.
"""
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

DATA = Path(__file__).resolve().parent / "data"
META = DATA / "capture_meta.csv"


def unit(P):
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)


def fit_projection(X, names, pca_dim=128, lda=False):
    """Return f(Z) -> unit vectors in the matcher's space."""
    pca = PCA(n_components=min(pca_dim, *X.shape), svd_solver="full", random_state=0).fit(X)
    if not lda:
        return lambda Z: unit(pca.transform(Z))
    l = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(pca.transform(X), names)
    return lambda Z: unit(l.transform(pca.transform(Z)))


def capture_days(capture_ids):
    """capture_id -> 'YYYY-MM-DD' from data/capture_meta.csv (fetch_meta.py); '' if unknown."""
    if not META.exists():
        return {c: "" for c in capture_ids}
    meta = {int(r["capture_id"]): r["captured_date"] for r in csv.DictReader(META.open())}
    return {c: meta.get(c, "") for c in capture_ids}


def turtle_day_groups(capture_ids, names):
    """Group id per photo: one sighting = same turtle + same day.

    box-turtle-id holds some photos twice under different dates (byte-identical
    image files, or the same original filename on two captures). Those are
    merged into one group so a photo can never count as its own "other day".
    Undated photos are their own group.
    """
    days = capture_days(capture_ids)
    meta = {}
    if META.exists():
        meta = {int(r["capture_id"]): r["original_filename"] for r in csv.DictReader(META.open())}
    parent = list(range(len(capture_ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_by(key_of):
        first = {}
        for i, (c, n) in enumerate(zip(capture_ids, names)):
            k = key_of(c, n)
            if k is None:
                continue
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i

    union_by(lambda c, n: (n, days[c]) if days[c] else None)
    union_by(lambda c, n: (n, meta[c]) if meta.get(c) else None)
    union_by(lambda c, n: _file_hash(n, c))
    return np.array([f"{names[find(i)]}|{capture_ids[find(i)]}" for i in range(len(capture_ids))])


def _file_hash(name, capture_id):
    p = DATA / name / f"{capture_id}.jpg"
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else None


def oof_sims(X, names, groups, pca_dim=128, lda=False, folds=5):
    """Similarity matrix where row i is scored by a projection fit without i's group."""
    names = np.asarray(names)
    if not lda:  # unsupervised: holding out changes little, fit once
        P = fit_projection(X, names, pca_dim)(X)
        return P @ P.T
    S = np.zeros((len(X), len(X)), np.float32)
    for tr, te in GroupKFold(folds).split(X, names, groups):
        P = fit_projection(X[tr], names[tr], pca_dim, lda=True)(X)
        S[te] = P[te] @ P.T
    return S


def calibrate(S, names, groups, keep_known=None):
    """Pick the 'new turtle' cut-off on similarity.

    For each reference photo (excluding its own turtle-day group):
      s_known = best match among *other days* of the same turtle
      s_new   = best match among *other turtles* (as if it were new)
    By default maximize the mean of P(s_known >= t) and P(s_new < t). With
    keep_known (e.g. 0.8), take the highest cut-off that still recognises that
    share of known turtles: across days the two piles overlap heavily, so the
    balanced cut-off would call about half of known turtles "new".
    """
    names, groups = np.asarray(names), np.asarray(groups)
    same_turtle = names[:, None] == names[None, :]
    same_group = groups[:, None] == groups[None, :]
    S = np.where(same_group, -np.inf, S)
    s_known = np.where(same_turtle, S, -np.inf).max(1)
    s_new = np.where(~same_turtle, S, -np.inf).max(1)
    s_known = s_known[np.isfinite(s_known)]
    cands = np.unique(np.concatenate([s_known, s_new]))
    kept = (s_known[None, :] >= cands[:, None]).mean(1)
    caught = (s_new[None, :] < cands[:, None]).mean(1)
    if keep_known is None:
        i = int(np.argmax(kept + caught))
    else:
        i = int(np.flatnonzero(kept >= keep_known).max())
    return float(cands[i]), {"known_kept": round(float(kept[i]), 3), "new_caught": round(float(caught[i]), 3)}


def rank(sim, names, paths, top=5, per_class=3):
    """Top `top` turtles by their best photo, with up to `per_class` reference photos each."""
    hits: dict[str, list] = {}
    for i in np.argsort(-sim):
        n = names[i]
        if n not in hits and len(hits) >= top:
            continue
        refs = hits.setdefault(n, [])
        if len(refs) < per_class:
            refs.append({"path": paths[i], "sim": round(float(sim[i]), 3)})
    return [{"name": n, "sim": refs[0]["sim"], "refs": refs} for n, refs in hits.items()]


def capture_id_of(path: str) -> int:
    return int(re.search(r"(\d+)\.jpg$", path).group(1))
