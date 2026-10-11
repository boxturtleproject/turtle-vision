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


def capture_meta(capture_ids):
    """capture_id -> its row of data/capture_meta.csv (fetch_meta.py); {} if unknown."""
    if not META.exists():
        return {c: {} for c in capture_ids}
    meta = {int(r["capture_id"]): r for r in csv.DictReader(META.open())}
    return {c: meta.get(c, {}) for c in capture_ids}


def turtle_day_groups(capture_ids, names):
    """Group id per photo: one sighting = same turtle + same encounter date.

    Photos in an encounter use the encounter's date from box-turtle-id. Filename
    dates are often the day a photo was edited, not taken: photos from 2020-2025
    encounters mostly carry 2025 filename dates, and one encounter's photos can
    carry several. So filename dates are used only for photos on a turtle's own
    record (no encounter): such a photo joins every photo of its turtle with the
    same filename date, which over-merges rather than risk a photo being scored
    against its own sighting.

    box-turtle-id also holds some pictures twice: byte-identical image files, the
    same original filename on two captures, or the same picture re-saved under
    another name (data/duplicate_pictures.csv, found by analysis/audit/label_audit.py).
    Those are merged too, so a photo can never count as its own "other day".
    Photos with no date at all are their own group.
    """
    meta = capture_meta(capture_ids)
    parent = list(range(len(capture_ids)))
    index = {c: i for i, c in enumerate(capture_ids)}

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

    enc_day = lambda c: meta[c].get("encounter_date", "")
    file_day = lambda c: meta[c].get("captured_date", "")
    union_by(lambda c, n: (n, "encounter", enc_day(c)) if enc_day(c) else None)
    by_file_day = {}
    for i, (c, n) in enumerate(zip(capture_ids, names)):
        if file_day(c):
            by_file_day.setdefault((n, file_day(c)), []).append(i)
    for i, (c, n) in enumerate(zip(capture_ids, names)):  # photos on the turtle's own record
        if not meta[c].get("encounter_id") and file_day(c):
            for j in by_file_day[(n, file_day(c))]:
                parent[find(j)] = find(i)
    union_by(lambda c, n: (n, meta[c]["original_filename"]) if meta[c].get("original_filename") else None)
    union_by(lambda c, n: _file_hash(n, c))
    for a, b in duplicate_pictures():  # re-saved copies (not byte-identical), from analysis/audit
        if a in index and b in index:
            parent[find(index[b])] = find(index[a])
    return np.array([f"{names[find(i)]}|{capture_ids[find(i)]}" for i in range(len(capture_ids))])


DUPLICATES = DATA / "duplicate_pictures.csv"


def duplicate_pictures():
    """(capture_id, capture_id) pairs that are the same picture saved twice."""
    if not DUPLICATES.exists():
        return []
    return [(int(r["capture_id_a"]), int(r["capture_id_b"])) for r in csv.DictReader(DUPLICATES.open())]


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


def fuse(sift_scores, emb_sims, cidx, n_classes, weight=1.0):
    """Per-turtle score: log(1 + best SIFT score) + weight * best embedding similarity.

    SIFT is decisive when spots match; the embedding similarity breaks ties
    when SIFT finds little. On the different-day test this beats either alone
    (RootSIFT + mutual matches: 0.843 top-1 vs 0.831 SIFT alone, 0.476
    embeddings; 515 queries grouped by encounter). Earlier measurements, with
    sightings grouped by filename date: weights 0.5-4 all landed within
    0.835-0.861; rank fusion was worse.
    """
    s = np.zeros(n_classes)
    np.maximum.at(s, cidx, np.where(np.isfinite(sift_scores), sift_scores, 0))
    e = np.full(n_classes, -np.inf)
    np.maximum.at(e, cidx, emb_sims)
    return np.log1p(s) + weight * np.where(np.isfinite(e), e, -1.0)

