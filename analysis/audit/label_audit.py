"""Audit the reference photos for misfiled and duplicate pictures (local data only).

Every reference shell crop is compared with every other one. A comparison counts
"consistent matches": mutual RootSIFT matches (ratio 0.8) that also fit one
stretch of the image (loose affine RANSAC). Two different turtles never exceed
6 in the reference set; repeat photos of the same turtle are usually 20 to 800.

Flags, per photo:
  misfiled   matches another turtle strongly (>= STRONG) and more than any
             photo of its own turtle
  duplicate  the same picture as another photo filed under a different turtle
             or date (checked by aligning the two originals: almost no pixel
             difference once aligned)
  no match   a top/left/right photo that matches none of its own turtle's
             photos (< WEAK), though the turtle has others of that view;
             a poor photo, or filed under the wrong turtle

Writes analysis/out/audit/pairs.npz (all pair counts) and flags.json; the review
page is built by review_page.py.

    python analysis/audit/label_audit.py
"""
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

TV = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TV), str(TV / "analysis/sift_study")]
import sift_match as sm  # noqa: E402
from lab_data import load_ref  # noqa: E402

OUT = TV / "analysis/out/audit"
STRONG = 15   # consistent matches: well above the most any two different turtles reached (6)
WEAK = 8      # below this a photo has no real support from its own turtle
DUP_PAIRS = 100  # pairs this strong are checked for being the same picture
DUP_DIFF = 10.0  # mean pixel difference after alignment below this = same picture


def feats(path):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    small = cv2.resize(img, (sm.WIDTH, max(1, round(h * sm.WIDTH / w))), interpolation=cv2.INTER_AREA)
    kp, d = sm._sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    if d is None:
        return np.zeros((0, 2), np.float32), None, np.float32(small.shape[1::-1])
    return (np.array([k.pt for k in kp], np.float32), sm.prepare((len(kp), d.astype(np.uint8)))[1],
            np.float32(small.shape[1::-1]))


F = None


def _init(paths):
    global F
    cv2.setNumThreads(1)  # one OpenCV thread per worker process
    F = [feats(p) for p in paths]


def _row(i):
    """(mutual, consistent) counts of photo i against photos j > i."""
    from lab_sift import ransac
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pa, da, sa = F[i]
    out = []
    for j in range(i + 1, len(F)):
        pb, db, sb = F[j]
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            out.append((0, 0))
            continue
        ab, ba = sm._nearest(bf, da, db), sm._nearest(bf, db, da)
        jj = ab[:, 0].astype(int)
        back = ba[jj]
        g = (jj >= 0) & (back[:, 0] == np.arange(len(da))) & (np.maximum(ab[:, 1], back[:, 1]) < sm.RATIO)
        k = np.flatnonzero(g)
        c = ransac(pa[k], pb[jj[k]], sa, sb, "affine_loose") if len(k) >= 5 else 0
        out.append((len(k), c))
    return i, out


def same_picture(path_a, path_b):
    """Mean pixel difference after aligning b onto a with SIFT + homography (None if no fit)."""
    a, b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
    side = 800
    a = cv2.resize(a, (side, round(a.shape[0] * side / a.shape[1])))
    b = cv2.resize(b, (side, round(b.shape[0] * side / b.shape[1])))
    s = cv2.SIFT_create(4000)
    ka, da = s.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = s.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None
    m = [x for x, y in (p for p in cv2.BFMatcher().knnMatch(db, da, k=2) if len(p) == 2) if x.distance < 0.7 * y.distance]
    if len(m) < 30:
        return None
    H, inl = cv2.findHomography(np.float32([kb[x.queryIdx].pt for x in m]), np.float32([ka[x.trainIdx].pt for x in m]),
                                cv2.RANSAC, 3.0)
    if H is None:
        return None
    w = cv2.warpPerspective(b, H, (a.shape[1], a.shape[0]))
    mask = cv2.warpPerspective(np.ones(b.shape[:2], np.uint8), H, (a.shape[1], a.shape[0])) > 0
    if mask.mean() < 0.5:  # mostly non-overlapping: not the same framing
        return None
    return float(np.abs(w.astype(np.int16) - a.astype(np.int16))[mask].mean())


def main():
    REF = load_ref()
    ids = [str(i) for i in REF["ids"]]
    names = list(REF["names"])
    crops = [Path(p) for p in REF["paths"]]
    meta = {r["capture_id"]: r for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
    orig = {p.stem: p for p in (TV / "data").glob("*/*.jpg") if p.parent.name not in ("crops", "crops_tight")}
    n = len(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "pairs.npz"
    if cache.exists() and list(np.load(cache)["ids"]) == ids:
        C = np.load(cache)["consistent"]
        M = np.load(cache)["mutual"]
    else:
        print(f"comparing {n} photos ({n * (n - 1) // 2} pairs) on 10 processes…", flush=True)
        C = np.zeros((n, n), np.int32)
        M = np.zeros((n, n), np.int32)
        with Pool(10, initializer=_init, initargs=(crops,)) as pool:
            for done, (i, row) in enumerate(pool.imap_unordered(_row, range(n), chunksize=4), 1):
                for k, (m, c) in enumerate(row):
                    j = i + 1 + k
                    M[i, j] = M[j, i] = m
                    C[i, j] = C[j, i] = c
                if done % 100 == 0:
                    print(f"  {done}/{n}", flush=True)
        np.savez_compressed(cache, ids=np.array(ids), consistent=C, mutual=M)

    date = [meta.get(i, {}).get("captured_date", "") for i in ids]
    view = [meta.get(i, {}).get("view", "") for i in ids]
    names_a = np.array(names)
    flags = []

    # duplicates: very strong pairs that are the same picture after alignment
    dup_of = {}
    cand = [(i, j) for i in range(n) for j in range(i + 1, n) if C[i, j] >= DUP_PAIRS]
    print(f"{len(cand)} very strong pairs to check for being the same picture", flush=True)
    for i, j in cand:
        if names[i] == names[j] and date[i] == date[j]:
            continue  # same turtle, same day: burst shots are expected
        d = same_picture(orig[ids[i]], orig[ids[j]])
        if d is not None and d < DUP_DIFF:
            flags.append({"kind": "duplicate", "photo": ids[i], "other": ids[j], "consistent": int(C[i, j]),
                          "pixel_diff": round(d, 1)})
            dup_of.setdefault(i, set()).add(j)
            dup_of.setdefault(j, set()).add(i)

    for i in range(n):
        others = np.ones(n, bool)
        others[i] = False
        for j in dup_of.get(i, ()):  # don't let an exact duplicate decide where a photo belongs
            others[j] = False
        own = others & (names_a == names[i])
        other = others & (names_a != names[i])
        own_best = int(C[i][own].max()) if own.any() else 0
        j_other = int(np.argmax(np.where(other, C[i], -1)))
        other_best = int(C[i, j_other])
        j_own = int(np.argmax(np.where(own, C[i], -1))) if own.any() else None
        rec = {"photo": ids[i], "turtle": names[i], "own_best": own_best, "own_photo": ids[j_own] if j_own is not None else None,
               "other_best": other_best, "other_photo": ids[j_other], "other_turtle": names[j_other]}
        if other_best >= STRONG and other_best > own_best:
            flags.append({"kind": "misfiled", **rec})
        elif (own_best < WEAK and view[i] in ("carapace_top", "carapace_left", "carapace_right")
              and any(view[k] == view[i] and date[k] != date[i] for k in np.flatnonzero(own))):
            flags.append({"kind": "no match", **rec})
    (OUT / "flags.json").write_text(json.dumps(flags, indent=1))
    # the duplicate pictures feed matching.turtle_day_groups (committed, unlike the rest)
    with (TV / "data/duplicate_pictures.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capture_id_a", "capture_id_b", "pixel_diff", "turtle_a", "turtle_b",
                    "filename_a", "filename_b"])
        for d in sorted((x for x in flags if x["kind"] == "duplicate"), key=lambda x: sorted((int(x["photo"]), int(x["other"])))):
            a, b = sorted((d["photo"], d["other"]), key=int)
            w.writerow([a, b, d["pixel_diff"], names[ids.index(a)], names[ids.index(b)],
                        meta.get(a, {}).get("original_filename", ""), meta.get(b, {}).get("original_filename", "")])
    from collections import Counter
    print(Counter(f["kind"] for f in flags))


if __name__ == "__main__":
    main()
