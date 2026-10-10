"""Evaluation data for SIFT experiments.

REF:   reference shell crops (data/crops/<name>/<id>.jpg) with turtle-day groups and
       out-of-fold combined (crop+tight LDA) embedding similarities.
FIELD: the verified field photos' shell crops (uploads/<uid>_crop.jpg), known + new,
       with embedding similarities to every reference (projection fit on all refs).
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TV))
import identify  # noqa: E402
import matching  # noqa: E402
import shellcrop  # noqa: E402

CACHE = TV / "analysis/out/sift_study"  # gitignored: features, score matrices, field embeddings
CACHE.mkdir(parents=True, exist_ok=True)


def load_ref():
    out = {}
    for v in ("crop", "tight"):
        names, paths, X = identify.load_reference(identify.DROP, TV / "data" / shellcrop.VARIANTS[v]["db"])
        out[v] = (names, [matching.capture_id_of(p) for p in paths], X)
    names, ids = out["crop"][0], out["crop"][1]
    assert out["tight"][1] == ids
    groups = matching.turtle_day_groups(ids, names)
    crops = [TV / "data" / shellcrop.VARIANTS["crop"]["dir"] / n / f"{c}.jpg" for n, c in zip(names, ids)]
    f = CACHE / "ref_emb_oof.npy"
    if f.exists():
        S = np.load(f)
    else:
        S = sum(matching.oof_sims(out[v][2], names, groups, 128, lda=True) for v in ("crop", "tight")) / 2
        np.save(f, S)
    return dict(names=np.array(names), ids=ids, groups=groups, paths=crops, S_emb=S, X=out)


def load_field(ref):
    rows = list(csv.DictReader((TV / "analysis/field_2026-10/uploads_summary.csv").open()))
    rows = [r for r in rows if r["true_answer"] and r["true_answer"] != "bad photo"]
    taken = {r["upload_id"]: r["taken"] for r in csv.DictReader((TV / "analysis/field_2026-10/photo_meta.csv").open())}
    uids = [r["upload_id"] for r in rows]
    truth = np.array([r["true_answer"] for r in rows])
    kind = np.array(["new" if t == "new turtle" else "known" for t in truth])
    day = [(taken.get(u) or r["time"])[:10] for u, r in zip(uids, rows)]
    paths = [TV / "uploads" / f"{u}_crop.jpg" for u in uids]
    f = CACHE / "field_emb.npz"
    vecs = dict(np.load(f)) if f.exists() else {}
    todo = [(u, v) for u in uids for v in ("crop", "tight") if f"{u}_{v}" not in vecs]
    if todo:
        identify.load_env()
        with ThreadPoolExecutor(8) as pool:
            res = list(pool.map(lambda uv: identify.embed_image(TV / "uploads" / f"{uv[0]}_{uv[1]}.jpg"), todo))
        for (u, v), e in zip(todo, res):
            vecs[f"{u}_{v}"] = e
        np.savez(f, **vecs)
    S = 0
    for v in ("crop", "tight"):
        names, _, X = ref["X"][v]
        proj = matching.fit_projection(X, np.array(names), 128, lda=True)
        P = proj(X)
        Q = proj(np.stack([vecs[f"{u}_{v}"] for u in uids]))
        S = S + Q @ P.T / 2
    return dict(uids=uids, truth=truth, kind=kind, day=day, paths=paths, S_emb=S)


if __name__ == "__main__":
    ref = load_ref()
    print(len(ref["ids"]), "refs", len(set(ref["names"])), "turtles")
    fld = load_field(ref)
    print(len(fld["uids"]), "field", (fld["kind"] == "known").sum(), "known")
    # sanity: embeddings-only top-1 on field known, expected ~47.7%
    classes = sorted(set(ref["names"]))
    cidx = np.array([classes.index(n) for n in ref["names"]])
    hit = 0
    for i in np.where(fld["kind"] == "known")[0]:
        best = np.full(len(classes), -np.inf)
        np.maximum.at(best, cidx, fld["S_emb"][i])
        hit += classes[int(np.argmax(best))] == fld["truth"][i]
    print("field embeddings-only top-1", hit / (fld["kind"] == "known").sum())
