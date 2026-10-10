"""SIFT feature variants and an exhaustive GPU matcher producing many per-reference scores.

For one feature config, every query (a REF crop or a FIELD crop) is matched against every
reference crop. Per (query, reference) we keep raw evidence:
  q2r / r2q ratio-test counts at several ratios, mutual-NN counts, RANSAC inliers
and per query the LNBNN (one-vs-many) scores. Scores are saved as matrices for lab_eval.py.

    python lab_sift.py '{"root": true}'        # 100-photo screening sample (~2 min)
    python lab_sift.py '{"root": true}' full   # every query (~20 min)

Config keys (BASE): width, pre (gray / clahe / lab_b / sat / norm), mask (None / ellipse),
root (RootSIFT), uint8, ct (contrast threshold), nfeat, sigma. Needs torch (MPS or CPU).
"""
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from lab_data import CACHE, load_field, load_ref

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RATIOS = (0.6, 0.67, 0.75, 0.8, 0.85, 0.9)
KEEP = 0.9  # candidate matches for RANSAC: q2r ratio below this

BASE = dict(width=250, pre="gray", mask=None, root=False, uint8=True, ct=0.04, nfeat=0, sigma=1.6)


def key(cfg):
    c = {**BASE, **cfg}
    return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]


def extract(path, cfg):
    c = {**BASE, **cfg}
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    W = c["width"]
    img = cv2.resize(img, (W, max(1, round(h * W / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if c["pre"] == "gray":
        g = gray
    elif c["pre"] == "clahe":
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    elif c["pre"] == "lab_b":  # yellow-blue axis: yellow spots light, brown/black dark
        g = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2]
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    elif c["pre"] == "sat":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    elif c["pre"] == "norm":  # per-image contrast stretch
        g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    mask = None
    if c["mask"] == "ellipse":
        H = g.shape[0]
        mask = np.zeros_like(g)
        cv2.ellipse(mask, (W // 2, H // 2), (int(W / 2.2 * 1.08), int(H / 2.2 * 1.08)), 0, 0, 360, 255, -1)
    sift = cv2.SIFT_create(nfeatures=c["nfeat"], contrastThreshold=c["ct"], sigma=c["sigma"])
    kp, desc = sift.detectAndCompute(g, mask)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), (g.shape[1], g.shape[0])
    if c["uint8"]:
        desc = desc.astype(np.uint8).astype(np.float32)
    if c["root"]:
        desc = np.sqrt(desc / (desc.sum(1, keepdims=True) + 1e-7))
    return np.array([k.pt for k in kp], np.float32), desc.astype(np.float32), (g.shape[1], g.shape[0])


def features(paths, cfg, tag):
    f = CACHE / f"feat_{tag}_{key(cfg)}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    with ThreadPoolExecutor(10) as pool:
        out = list(pool.map(lambda p: extract(p, cfg), paths))
    f.write_bytes(pickle.dumps(out))
    return out


class RefBank:
    def __init__(self, feats):
        self.n = len(feats)
        self.lens = np.array([len(f[1]) for f in feats])
        M = max(1, self.lens.max())
        R = np.zeros((self.n, M, 128), np.float32)
        for k, f in enumerate(feats):
            R[k, : len(f[1])] = f[1]
        self.R = torch.from_numpy(R).to(DEV)
        self.Rn = (self.R ** 2).sum(-1)  # (n, M)
        valid = torch.arange(M)[None, :] < torch.from_numpy(self.lens)[:, None]
        self.valid = valid.to(DEV)
        self.xy = [f[0] for f in feats]
        self.size = [f[2] for f in feats]


def _top2(G, dim):
    """Largest two along dim of G (bigger = closer) and the argmax."""
    g1, j = G.max(dim, keepdim=True)
    g2 = G.scatter(dim, j, float("-inf")).max(dim).values
    return g1.squeeze(dim), g2, j.squeeze(dim)


def match_query(bank: RefBank, q, chunk=64):
    """For one query's descriptors: per ref, q2r nn/ratio (n,) and r2q nn/ratio (M,)."""
    Q = torch.from_numpy(q).to(DEV)
    Qn = (Q ** 2).sum(-1)
    n = len(q)
    qj, qd1, qd2, ri, rr = [], [], [], [], []
    for s in range(0, bank.n, chunk):
        R, Rn, V = bank.R[s : s + chunk], bank.Rn[s : s + chunk], bank.valid[s : s + chunk]
        # D = Qn + Rn - 2QR; G = 2QR - Rn so that D = Qn - G
        G = 2 * torch.einsum("nd,bmd->bnm", Q, R) - Rn[:, None, :]
        G = torch.where(V[:, None, :], G, torch.full_like(G, float("-inf")))
        g1, g2, j = _top2(G, 2)
        qj.append(j)
        qd1.append((Qn[None] - g1).clamp_min(0).sqrt())
        qd2.append((Qn[None] - g2).clamp_min(0).sqrt())
        # r2q: D = Rn + Qn - 2QR ; H = 2QR - Qn
        Hm = G + Rn[:, None, :] - Qn[None, :, None]
        h1, h2, i = _top2(Hm, 1)
        d1 = (Rn - h1).clamp_min(0).sqrt()
        d2 = (Rn - h2).clamp_min(0).sqrt()
        ri.append(i)
        rr.append(d1 / d2.clamp_min(1e-9))
    q_j = torch.cat(qj).cpu().numpy()
    q_d1 = torch.cat(qd1).cpu().numpy()
    q_d2 = torch.cat(qd2).cpu().numpy()
    ri = torch.cat(ri).cpu().numpy()
    rr = torch.cat(rr).cpu().numpy()
    if n < 2:
        rr[:] = 1
    r_i = [ri[k, : bank.lens[k]] for k in range(bank.n)]
    r_ratio = [rr[k, : bank.lens[k]] for k in range(bank.n)]
    return q_j, q_d1, q_d2, r_i, r_ratio


def ransac(pq, pr, size_q, size_r, method):
    if len(pq) < 5:
        return 0
    # coordinates in units of image width so thresholds are resolution free
    a = pq / size_q[0]
    b = pr / size_r[0]
    try:
        if method == "homog":
            if len(pq) < 5:
                return 0
            H, m = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 0.02, maxIters=2000, confidence=0.999)
        elif method == "affine":
            H, m = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=0.03,
                                               maxIters=2000, confidence=0.99)
        elif method == "affine_loose":
            H, m = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=0.06,
                                        maxIters=2000, confidence=0.99)
        elif method == "fund":
            H, m = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, 0.01, 0.999, 2000)
    except cv2.error:
        return 0
    return int(m.sum()) if m is not None else 0


def lnbnn(q_d1, names_cidx, n_classes, excl, K=5):
    """HotSpotter-style one-vs-many score per class and per ref.

    q_d1: (refs, n) nearest distance of each query descriptor within each ref.
    Global neighbours approximated by each ref's nearest descriptor (one vote per ref).
    """
    D = np.where(excl[:, None], np.inf, q_d1).T  # (n, refs)
    idx = np.argpartition(D, K, axis=1)[:, : K + 1]
    dd = np.take_along_axis(D, idx, 1)
    o = np.argsort(dd, 1)
    idx, dd = np.take_along_axis(idx, o, 1), np.take_along_axis(dd, o, 1)
    norm = dd[:, K : K + 1]
    w = np.clip(norm - dd[:, :K], 0, None)  # (n, K)
    per_ref = np.zeros(q_d1.shape[0])
    np.add.at(per_ref, idx[:, :K].ravel(), w.ravel())
    per_cls = np.zeros(n_classes)
    # one vote per class per query feature: its nearest ref of that class among the K
    c = names_cidx[idx[:, :K]]
    for k in range(K):
        first = np.ones(len(c), bool)
        for kk in range(k):
            first &= c[:, kk] != c[:, k]
        np.add.at(per_cls, c[first, k], w[first, k])
    return per_ref, per_cls


def run(cfg, tag="", ransac_methods=("affine",), ransac_min=6, limit=None, sample=False):
    t0 = time.time()
    ref = load_ref()
    fld = load_field(ref)
    rfeat = features(ref["paths"], cfg, "ref")
    ffeat = features(fld["paths"], cfg, "field")
    out_f = CACHE / f"scores_{key(cfg)}{'_s100' if sample else ''}.pkl"
    if out_f.exists():
        return pickle.loads(out_f.read_bytes())
    bank = RefBank(rfeat)
    names = ref["names"]
    classes = sorted(set(names))
    cidx = np.array([classes.index(x) for x in names])
    groups = ref["groups"]
    nref = len(names)
    kp_ref = bank.lens
    pool = ThreadPoolExecutor(10)

    def one(qf, excl_sets):
        xy_q, dq, size_q = qf
        n = len(dq)
        res = {"nq": n}
        if n < 2:
            z = np.zeros(nref)
            for r in RATIOS:
                res[f"q2r@{r}"] = z
                res[f"r2q@{r}"] = z
                res[f"mut@{r}"] = z
            res["rs_mutaff"] = z
            for m in ransac_methods:
                res[f"rs_{m}"] = z
                res[f"rs67_{m}"] = z
            for e in excl_sets:
                res[f"lnbnn_ref_{e}"] = z
                res[f"lnbnn_cls_{e}"] = np.zeros(len(classes))
            return res
        q_j, q_d1, q_d2, r_i, r_ratio = match_query(bank, dq)
        q_ratio = q_d1 / np.maximum(q_d2, 1e-9)
        for r in RATIOS:
            res[f"q2r@{r}"] = (q_ratio < r).sum(1).astype(np.float32)
            res[f"r2q@{r}"] = np.array([(rr < r).sum() for rr in r_ratio], np.float32)
        mut = {r: np.zeros(nref, np.float32) for r in RATIOS}
        cand = []
        for k in range(nref):
            js = q_j[k]
            ok = r_i[k][js] == np.arange(n)  # mutual NN
            rr = np.maximum(q_ratio[k], r_ratio[k][js])
            for r in RATIOS:
                mut[r][k] = (ok & (rr < r)).sum()
            sel = np.flatnonzero(q_ratio[k] < KEEP)
            cand.append((sel, js[sel], q_ratio[k][sel], (ok & (rr < 0.8))[sel]))
        for r in RATIOS:
            res[f"mut@{r}"] = mut[r]

        def rs(k):
            sel, js, rat, mm = cand[k]
            out = {}
            out["mut_aff"] = ransac(xy_q[sel[mm]], bank.xy[k][js[mm]], size_q, bank.size[k], "affine_loose") if mm.sum() >= 5 else 0
            for m in ransac_methods:
                rat_ok = rat < 0.8
                sel_, js_ = sel[rat_ok], js[rat_ok]
                if len(sel_) < ransac_min:
                    out[m] = out[m + "67"] = 0
                    continue
                out[m] = ransac(xy_q[sel_], bank.xy[k][js_], size_q, bank.size[k], m)
                s67 = rat < 0.67
                out[m + "67"] = ransac(xy_q[sel[s67]], bank.xy[k][js[s67]], size_q, bank.size[k], m) if s67.sum() >= 5 else 0
            return out

        rsl = list(pool.map(rs, range(nref)))
        for m in ransac_methods:
            res[f"rs_{m}"] = np.array([x[m] for x in rsl], np.float32)
            res[f"rs67_{m}"] = np.array([x[m + "67"] for x in rsl], np.float32)
        res["rs_mutaff"] = np.array([x["mut_aff"] for x in rsl], np.float32)
        for e, excl in excl_sets.items():
            pr, pc = lnbnn(q_d1, cidx, len(classes), excl)
            res[f"lnbnn_ref_{e}"] = pr
            res[f"lnbnn_cls_{e}"] = pc
        return res

    same_group = groups[:, None] == groups[None, :]
    same_turtle = names[:, None] == names[None, :]
    has_other = (same_turtle & ~same_group).any(1)
    ref_q = np.where(has_other)[0][:limit] if limit else np.where(has_other)[0]
    field_idx = np.arange(len(ffeat))[: limit or None]
    if sample:  # fixed screening sample: 50 reference queries + 30 known and 20 new field photos
        rng = np.random.default_rng(0)
        ref_q = np.sort(rng.choice(np.where(has_other)[0], 50, replace=False))
        field_idx = np.sort(np.concatenate([
            rng.choice(np.where(fld["kind"] == "known")[0], 30, replace=False),
            rng.choice(np.where(fld["kind"] == "new")[0], 20, replace=False)]))
    out = {"ref_q": ref_q, "field_idx": field_idx, "ref": [], "field": [], "kp_ref": kp_ref,
           "kp_field": np.array([len(f[1]) for f in ffeat])}
    for c, i in enumerate(ref_q):
        out["ref"].append(one(rfeat[i], {"grp": same_group[i], "tur": same_turtle[i]}))
        if c % 100 == 0:
            print(f"  ref {c}/{len(ref_q)} {time.time() - t0:.0f}s", flush=True)
    for c, fi in enumerate(field_idx):
        qf = ffeat[fi]
        out["field"].append(one(qf, {"none": np.zeros(nref, bool)}))
        if c % 100 == 0:
            print(f"  field {c}/{len(ffeat)} {time.time() - t0:.0f}s", flush=True)
    if not limit:
        out_f.write_bytes(pickle.dumps(out))
    print(f"done {cfg} in {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    full = len(sys.argv) > 2 and sys.argv[2] == "full"
    run(cfg, sample=not full)
