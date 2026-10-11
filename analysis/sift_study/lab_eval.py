"""Turn lab_sift score files into a comparison table.

Columns
  REF  (different-day test, 534 queries)   sift top1/top5, fused top1/top5, newAUC
  FIELD (176 known + 142 new)               sift top1/top5, fused top1/top5, newAUC, named@90
newAUC: how well the pick's score separates known from new (REF: the query with its own
turtle removed plays "new"). named@90: share of known photos named correctly when the
spot cut-off is set so 90% of new turtles fall below it.
"""
import json
import pickle
import sys

import numpy as np

from lab_data import CACHE, load_field, load_ref
from lab_sift import key

REF = load_ref()
FLD = load_field(REF)
NAMES = REF["names"]
CLASSES = sorted(set(NAMES))
CIDX = np.array([CLASSES.index(n) for n in NAMES])
K = len(CLASSES)
GROUPS = REF["groups"]


def score_vec(res, kind, kp_ref, excl_tag):
    """Per-reference score vector (or per-class with is_cls=True)."""
    nq = res["nq"]
    fewest = np.maximum(np.minimum(nq, kp_ref), 1)
    if kind.startswith("pre:"):
        return res[kind], False
    if kind == "base":
        return np.maximum(res["q2r@0.67"], res["r2q@0.67"]) / fewest * 100, False
    if kind.startswith("raw@"):
        r = kind[4:]
        return np.maximum(res[f"q2r@{r}"], res[f"r2q@{r}"]), False
    if kind.startswith("base@"):
        r = kind[5:]
        return np.maximum(res[f"q2r@{r}"], res[f"r2q@{r}"]) / fewest * 100, False
    if kind.startswith("mut@"):
        return res[kind], False
    if kind.startswith("mutn@"):
        return res["mut@" + kind[5:]] / fewest * 100, False
    if kind.startswith("mutg@"):  # geometric-mean normalisation
        return res["mut@" + kind[5:]] / np.sqrt(np.maximum(nq * kp_ref, 1)) * 100, False
    if kind.startswith("rs"):
        if kind.endswith("_n"):
            return res[kind[:-2]] / fewest * 100, False
        return res[kind], False
    if kind == "lnbnn_ref":
        return res[f"lnbnn_ref_{excl_tag}"], False
    if kind == "lnbnn_cls":
        return res[f"lnbnn_cls_{excl_tag}"], True
    raise KeyError(kind)


def per_class(v, is_cls, excl, agg="max"):
    if is_cls:
        return v.copy()
    v = np.where(excl, -np.inf, v)
    out = np.full(K, -np.inf)
    if agg == "max":
        np.maximum.at(out, CIDX, v)
    elif agg == "top2":
        for c in range(K):
            s = np.sort(v[CIDX == c])[::-1]
            s = s[np.isfinite(s)]
            out[c] = s[:2].sum() if len(s) else -np.inf
    return out


def emb_class(s, excl):
    e = np.full(K, -np.inf)
    np.maximum.at(e, CIDX, np.where(excl, -np.inf, s))
    return e


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def fuse(sc, ec, w):
    sc = np.where(np.isfinite(sc), np.maximum(sc, 0), 0)
    return w * np.log1p(sc) + np.where(np.isfinite(ec), ec, -1.0)


def evaluate(S, kind, agg="max", weights=(0.25, 0.5, 1, 2, 4)):
    kp_ref = S["kp_ref"]
    out = {}
    # ---- REF different-day
    rows = []
    for res, q in zip(S["ref"], S["ref_q"]):
        excl_g = GROUPS == GROUPS[q]
        excl_t = NAMES == NAMES[q]
        v, is_cls = score_vec(res, kind, kp_ref, "grp")
        sc = per_class(v, is_cls, excl_g, agg)
        v2, _ = score_vec(res, kind, kp_ref, "tur")
        sc_new = per_class(v2, is_cls, excl_t, agg)
        ec = emb_class(REF["S_emb"][q], excl_g)
        rows.append((CIDX[q], sc, sc_new, ec))
    n = len(rows)
    out["_ref_sift_ok"] = np.array([np.argmax(sc) == c for c, sc, _, _ in rows])
    out["ref_t1"] = out["_ref_sift_ok"].mean()
    out["ref_t5"] = np.mean([c in np.argsort(-sc)[:5] for c, sc, _, _ in rows])
    best = None
    for w in weights:
        t1 = np.mean([np.argmax(fuse(sc, ec, w)) == c for c, sc, _, ec in rows])
        if best is None or t1 > best[1]:
            best = (w, t1)
    w = best[0]
    out["w"] = w
    out["ref_f1"] = best[1]
    out["_ref_fused_ok"] = np.array([np.argmax(fuse(sc, ec, w)) == c for c, sc, _, ec in rows])
    out["ref_f5"] = np.mean([c in np.argsort(-fuse(sc, ec, w))[:5] for c, sc, _, ec in rows])
    # spot of the fused pick: known (own turtle allowed) vs as-new (own turtle removed)
    spot_k = [sc[np.argmax(fuse(sc, ec, w))] for c, sc, _, ec in rows]
    spot_n = []
    for c, sc, sc_new, ec in rows:
        ec2 = ec.copy()
        ec2[c] = -np.inf
        spot_n.append(sc_new[np.argmax(fuse(np.where(np.arange(K) == c, -np.inf, sc_new), ec2, w))])
    out["ref_auc"] = auc(spot_k, spot_n)
    # ---- FIELD
    frows = []
    for i, res in zip(S["field_idx"], S["field"]):
        v, is_cls = score_vec(res, kind, kp_ref, "none")
        sc = per_class(v, is_cls, np.zeros(len(NAMES), bool), agg)
        ec = emb_class(FLD["S_emb"][i], np.zeros(len(NAMES), bool))
        frows.append((FLD["kind"][i], FLD["truth"][i], sc, ec))
    known = [r for r in frows if r[0] == "known"]
    new = [r for r in frows if r[0] == "new"]
    ci = lambda t: CLASSES.index(t) if t in CLASSES else -1
    out["_fld_sift_ok"] = np.array([np.argmax(sc) == ci(t) for _, t, sc, _ in known])
    out["fld_t1"] = out["_fld_sift_ok"].mean()
    out["fld_t5"] = np.mean([ci(t) in np.argsort(-sc)[:5] for _, t, sc, _ in known])
    out["_fld_fused_ok"] = np.array([np.argmax(fuse(sc, ec, w)) == ci(t) for _, t, sc, ec in known])
    out["fld_f1"] = out["_fld_fused_ok"].mean()
    out["fld_f5"] = np.mean([ci(t) in np.argsort(-fuse(sc, ec, w))[:5] for _, t, sc, ec in known])
    sk = np.array([sc[np.argmax(fuse(sc, ec, w))] for _, _, sc, ec in known])
    sn = np.array([sc[np.argmax(fuse(sc, ec, w))] for _, _, sc, ec in new])
    right = np.array([np.argmax(fuse(sc, ec, w)) == ci(t) for _, t, sc, ec in known])
    out["fld_auc"] = auc(sk, sn)
    T = np.quantile(sn, 0.9)
    out["named90"] = np.mean((sk > T) & right)
    out["wrong90"] = np.mean((sk > T) & ~right)
    return out


COLS = ["ref_t1", "ref_t5", "ref_f1", "ref_f5", "ref_auc", "fld_t1", "fld_t5", "fld_f1", "fld_f5", "fld_auc", "named90", "wrong90", "w"]


def table(entries):
    print(f"{'config':34s} {'score':14s} " + " ".join(f"{c:>7s}" for c in COLS))
    for label, S, kind, agg in entries:
        o = evaluate(S, kind, agg)
        print(f"{label:34s} {kind + ('' if agg == 'max' else '/' + agg):14s} " + " ".join(f"{o[c]:7.3f}" for c in COLS), flush=True)


def combine(S1, S2, kind, w2=1.0):
    """Add one score kind from two configs (same queries): e.g. grey + yellow-channel SIFT."""
    out = {k: S1[k] for k in ("ref_q", "field_idx", "kp_ref", "kp_field")}
    tag = f"pre:{kind}"
    for part in ("ref", "field"):
        rows = []
        for r1, r2 in zip(S1[part], S2[part]):
            r = {"nq": r1["nq"]}
            v1, _ = score_vec(r1, kind, S1["kp_ref"], "grp" if part == "ref" else "none")
            v2, _ = score_vec(r2, kind, S2["kp_ref"], "grp" if part == "ref" else "none")
            r[tag] = v1 + w2 * v2  # count-based kinds only (same vector serves "as new")
            rows.append(r)
        out[part] = rows
    return out, tag


def load(cfg, full=False):
    S = pickle.loads((CACHE / f"scores_{key(cfg)}{'' if full else '_s100'}.pkl").read_bytes())
    # keep only queries that still have another sighting of their turtle under the current
    # groups (merging re-saved duplicate pictures leaves some photos with none)
    keep = [k for k, q in enumerate(S["ref_q"]) if ((NAMES == NAMES[q]) & (GROUPS != GROUPS[q])).any()]
    S["ref"], S["ref_q"] = [S["ref"][k] for k in keep], [S["ref_q"][k] for k in keep]
    return S


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    kinds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base"]
    S = load(cfg)
    table([(json.dumps(cfg) or "{}", S, k, "max") for k in kinds])
