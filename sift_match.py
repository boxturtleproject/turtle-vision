"""Spot matching with box-turtle-id's SIFT matcher.

Port of box-turtle-id backend/app/services/sift.py: SIFT keypoints on an
image resized to 250px wide, Lowe ratio test at 0.67 in both directions,
score = good matches / min(keypoint counts) * 100, and a match at >= 4.
Exact brute-force matching replaces FLANN (same pairs, no index to build).

Here it runs on the Gemini shell crop and compares against every reference
crop, any view. Different-day test (534 queries, evaluate.py --sift):

                        top-1  top-5  known turtles confirmed (score >= 4)
  shell crop            0.811  0.875  56%
  whole photo           0.788  0.867  41%
  + embeddings (fuse)   0.861 / 0.837 top-1 for crop / whole photo

Cropping costs the ~1.3s box call but confirms many more known turtles:
without background, more matched spots are on the shell.

Score >= 4 is strong evidence: the best *wrong* turtle reaches it for ~2%
of queries, the right turtle for ~56%. So "nothing scored 4+" flags most new
turtles, at the cost of leaving many known turtles unconfirmed.
"""
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

WIDTH = 250
RATIO = 0.67
CONFIRM = 4.0  # box-turtle-id's acceptance_threshold

_sift = cv2.SIFT_create()
_bf = cv2.BFMatcher(cv2.NORM_L2)


def features(path):
    """(keypoint count, uint8 descriptors) for an image file, resized to WIDTH."""
    img = cv2.imread(str(path))
    if img is None:
        return 0, None
    h, w = img.shape[:2]
    img = cv2.resize(img, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    kp, desc = _sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    return (len(kp), desc.astype(np.uint8)) if desc is not None else (0, None)


def _good(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0
    return sum(1 for m in _bf.knnMatch(a, b, k=2) if len(m) == 2 and m[0].distance < RATIO * m[1].distance)


def score(fa, fb):
    a = fa[1].astype(np.float32) if fa[1] is not None else None
    b = fb[1].astype(np.float32) if fb[1] is not None else None
    good = max(_good(a, b), _good(b, a))
    fewest = min(fa[0], fb[0])
    return good / fewest * 100 if fewest else 0.0


class SiftIndex:
    """SIFT features for every reference crop, cached on disk by capture id."""

    def __init__(self, names, ids, paths, cache: Path, root: Path = Path("."), workers=8):
        """paths: reference crops relative to root (also used for display)."""
        self.names, self.ids, self.paths = list(names), list(ids), list(paths)
        self.classes = sorted(set(self.names))
        self.threshold = CONFIRM
        self.calib = {"rule": f"spot-match score >= {CONFIRM:g} confirms a known turtle"}
        self.pool = ThreadPoolExecutor(workers)  # cv2 releases the GIL while matching
        cached = pickle.loads(cache.read_bytes()) if cache.exists() else {}
        missing = [i for i, c in enumerate(self.ids) if c not in cached]
        for i, f in zip(missing, self.pool.map(lambda i: features(root / self.paths[i]), missing)):
            cached[self.ids[i]] = f
        if missing:
            cache.write_bytes(pickle.dumps(cached))
        self.feats = [cached[c] for c in self.ids]

    def scores(self, query_feats, exclude=None):
        """Score against every reference; exclude: optional boolean mask to skip."""
        idx = [j for j in range(len(self.ids)) if exclude is None or not exclude[j]]
        out = np.full(len(self.ids), -np.inf)
        out[idx] = list(self.pool.map(lambda j: score(query_feats, self.feats[j]), idx))
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
