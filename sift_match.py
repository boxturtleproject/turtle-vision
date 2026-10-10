"""Spot matching: SIFT on the shell crop, mutual matches of RootSIFT descriptors.

Started as a port of box-turtle-id's matcher (backend/app/services/sift.py):
SIFT keypoints on an image resized to 250px wide, Lowe ratio test at 0.67,
score = good matches in the better direction / fewer keypoints * 100.

Improved after the SIFT study (analysis/SIFT_STUDY.md), same keypoints and cost:
  - RootSIFT: descriptors L1-normalised then square-rooted (less sensitive to
    lighting; HotSpotter uses it)
  - mutual matches: a pair counts only when each descriptor is the other's
    nearest neighbour, both passing the ratio test at 0.8 (looser than 0.67 is
    safe only with the mutual check)
  - score = mutual matches / sqrt(keypoints in both) * 100

Full-set results (SIFT + embeddings, matching.fuse):
                                     old      new
  field photos, right first time    90.3%    96.0%   (176 known turtles)
  different-day test, right first   85.2%    88.6%   (534 queries)
  SIFT alone, different-day test    81.5%    87.6%

Spot-score bands (the top turtle's score, new scale):
  >= 2.0  match         field: 127 known all right, no new turtles
  1.2-2   check         field: 22 known (all right) and 5 new
  < 1.2   possibly new  catches 96.5% of new turtles
With 2+ photos the check zone starts at 1.5: a new turtle's best photo of
several more often reaches 1.2 by chance.
"""
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

WIDTH = 250
RATIO = 0.8
MATCH, CHECK, CHECK_MULTI = 2.0, 1.2, 1.5

_sift = cv2.SIFT_create()


def band_of(spot, n_photos=1):
    """'match' / 'check' / 'possibly new' for the top turtle's spot score."""
    if spot >= MATCH:
        return "match"
    return "check" if spot >= (CHECK if n_photos == 1 else CHECK_MULTI) else "possibly new"


def features(path):
    """(keypoint count, uint8 descriptors) for an image file, resized to WIDTH."""
    img = cv2.imread(str(path))
    if img is None:
        return 0, None
    h, w = img.shape[:2]
    img = cv2.resize(img, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    kp, desc = _sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    return (len(kp), desc.astype(np.uint8)) if desc is not None else (0, None)


def prepare(f):
    """(keypoint count, RootSIFT float32 descriptors) from features()."""
    n, desc = f
    if desc is None:
        return n, None
    d = desc.astype(np.float32)
    return n, np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))


def _nearest(bf, a, b):
    """Per row of a: (index of nearest row of b, distance ratio to the second nearest)."""
    out = np.full((len(a), 2), [-1, 1.0])
    for m in bf.knnMatch(a, b, k=2):
        if len(m) == 2:
            out[m[0].queryIdx] = m[0].trainIdx, m[0].distance / max(m[1].distance, 1e-9)
    return out


def mutual_score(pa, pb):
    """Score two prepare()d features: mutual ratio-test matches / sqrt(n_a * n_b) * 100."""
    a, b = pa[1], pb[1]
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)  # one per call: matchers aren't shared across threads
    ab, ba = _nearest(bf, a, b), _nearest(bf, b, a)
    j = ab[:, 0].astype(int)
    back = ba[j]
    good = (j >= 0) & (back[:, 0] == np.arange(len(a))) & (np.maximum(ab[:, 1], back[:, 1]) < RATIO)
    return good.sum() / np.sqrt(pa[0] * pb[0]) * 100


def score(fa, fb):
    """Spot score between two features() results."""
    return mutual_score(prepare(fa), prepare(fb))


class SiftIndex:
    """SIFT features for every reference crop, cached on disk by capture id."""

    def __init__(self, names, ids, paths, cache: Path, root: Path = Path("."), workers=8):
        """paths: reference crops relative to root (also used for display)."""
        self.names, self.ids, self.paths = list(names), list(ids), list(paths)
        self.classes = sorted(set(self.names))
        self.threshold = CHECK
        self.calib = {"rule": f"spot score >= {MATCH:g} match, {CHECK:g}-{MATCH:g} check, below {CHECK:g} possibly new"}
        self.pool = ThreadPoolExecutor(workers)  # cv2 releases the GIL while matching
        cached = pickle.loads(cache.read_bytes()) if cache.exists() else {}
        missing = [i for i, c in enumerate(self.ids) if c not in cached]
        for i, f in zip(missing, self.pool.map(lambda i: features(root / self.paths[i]), missing)):
            cached[self.ids[i]] = f
        if missing:
            cache.write_bytes(pickle.dumps(cached))
        self.feats = [cached[c] for c in self.ids]
        self.prepped = [prepare(f) for f in self.feats]

    def scores(self, query_feats, exclude=None):
        """Score against every reference; exclude: optional boolean mask to skip."""
        idx = [j for j in range(len(self.ids)) if exclude is None or not exclude[j]]
        out = np.full(len(self.ids), -np.inf)
        q = prepare(query_feats)
        out[idx] = list(self.pool.map(lambda j: mutual_score(q, self.prepped[j]), idx))
        return out

    def query_scores(self, path):
        """Score an image file against every reference photo."""
        return self.scores(features(path))

    def query(self, path, top=5, per_class=3):
        return self.rank(self.query_scores(path), top, per_class)

    def rank(self, s, top=5, per_class=3):
        hits: dict[str, list] = {}
        for j in np.argsort(-s):
            n = self.names[j]
            if n not in hits and len(hits) >= top:
                continue
            refs = hits.setdefault(n, [])
            if len(refs) < per_class:
                refs.append({"path": self.paths[j], "sim": round(float(s[j]), 1)})
        return [{"name": n, "sim": refs[0]["sim"], "refs": refs} for n, refs in hits.items()]
