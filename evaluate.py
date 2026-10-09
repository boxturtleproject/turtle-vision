"""Compare matchers on the honest "different day" test.

Every reference photo is a query. It may only match photos of its turtle
taken on *other days* (same-day siblings share background, light and pose,
which inflates scores). Photos whose turtle has no other dated day are
skipped. LDA matchers are scored out-of-fold (see matching.oof_sims).

Needs data/embeddings*.sqlite (embed_photos.py, crop_photos.py) and
data/capture_meta.csv (fetch_meta.py).

    python evaluate.py
"""
import numpy as np

import identify
import matching
import shellcrop

DBS = {"full": identify.DB, **{v: identify.DATA / cfg["db"] for v, cfg in shellcrop.VARIANTS.items()}}


def topk_hits(S, names, groups, k=5):
    names, groups = np.asarray(names), np.asarray(groups)
    classes = sorted(set(names))
    cidx = np.array([classes.index(n) for n in names])
    same_group = groups[:, None] == groups[None, :]
    has_other_day = ((names[:, None] == names[None, :]) & ~same_group).any(1)
    t1 = t5 = 0
    queries = np.where(has_other_day)[0]
    for i in queries:
        s = np.where(same_group[i], -np.inf, S[i])
        best = np.full(len(classes), -np.inf)
        np.maximum.at(best, cidx, s)
        top = np.argsort(-best)[:k]
        t1 += cidx[i] == top[0]
        t5 += cidx[i] in top
    return t1 / len(queries), t5 / len(queries), len(queries)


def main():
    refs = {}
    for v, db in DBS.items():
        if db.exists():
            names, paths, X = identify.load_reference(identify.DROP, db)
            refs[v] = (names, [matching.capture_id_of(p) for p in paths], X)
    first = next(iter(refs.values()))
    names, ids = first[0], first[1]
    assert all(r[1] == ids for r in refs.values()), "embedding DBs cover different photos"
    groups = matching.turtle_day_groups(ids, names)

    print(f"{len(ids)} reference photos, {len(set(names))} turtles\n")
    print(f"{'matcher':28s} {'top-1':>6s} {'top-5':>6s}   new-turtle cut-off")
    sims = {}
    for v, (_, _, X) in refs.items():
        for lda in (False, True):
            S = matching.oof_sims(X, names, groups, identify.DEFAULT_PCA, lda)
            sims[(v, lda)] = S
            t1, t5, n = topk_hits(S, names, groups)
            thr, cal = matching.calibrate(S, names, groups)
            print(f"{v + (' LDA' if lda else ' PCA'):28s} {t1:6.3f} {t5:6.3f}   "
                  f"{thr:.3f} (keeps {cal['known_kept']:.0%} known, flags {cal['new_caught']:.0%} new)")
    if ("crop", True) in sims and ("tight", True) in sims:
        S = (sims[("crop", True)] + sims[("tight", True)]) / 2
        t1, t5, n = topk_hits(S, names, groups)
        thr, cal = matching.calibrate(S, names, groups)
        print(f"{'combined: crop+tight LDA':28s} {t1:6.3f} {t5:6.3f}   "
              f"{thr:.3f} (keeps {cal['known_kept']:.0%} known, flags {cal['new_caught']:.0%} new)")
    print(f"\n({n} queries with another dated day of the same turtle)")


if __name__ == "__main__":
    main()
