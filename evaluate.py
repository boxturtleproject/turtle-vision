"""Compare matchers on the honest "different day" test.

Every reference photo is a query. It may only match photos of its turtle
taken on *other days* (same-day siblings share background, light and pose,
which inflates scores). Photos whose turtle has no other dated day are
skipped. LDA matchers are scored out-of-fold (see matching.oof_sims).

Needs data/embeddings*.sqlite (embed_photos.py, crop_photos.py) and
data/capture_meta.csv (fetch_meta.py).

    python evaluate.py
    python evaluate.py --gemini 120   # also Gemini re-ranking on 120 random
                                      # queries (~$0.005 and ~9k tokens each)
    python evaluate.py --sift         # also box-turtle-id's SIFT spot matcher
                                      # against every reference crop (~5-25 min)
"""
import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import identify
import matching
import rerank
import shellcrop
import sift_match

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


def gemini_eval(S, names, ids, groups, n_queries):
    """Re-rank the combined top 5 with Gemini for a random sample of queries."""
    names, groups = np.asarray(names), np.asarray(groups)
    same_group = groups[:, None] == groups[None, :]
    queries = [int(i) for i in np.where(((names[:, None] == names[None, :]) & ~same_group).any(1))[0]]
    random.Random(0).shuffle(queries)
    queries = queries[:n_queries]
    crop = lambda i: identify.DATA / shellcrop.VARIANTS["crop"]["dir"] / names[i] / f"{ids[i]}.jpg"
    client = shellcrop.make_client()

    def one(q):
        s = np.where(same_group[q], -np.inf, S[q])
        cands = {}
        for j in np.argsort(-s):
            if not np.isfinite(s[j]) or (len(cands) == 5 and names[j] not in cands):
                continue
            refs = cands.setdefault(names[j], [])
            if len(refs) < 3:
                refs.append(crop(j))
        base = list(cands)[0] == names[q]
        try:
            pick = rerank.rerank(client, crop(q), cands, seed=q)["ranking"][0] == names[q]
        except Exception as e:
            print(f"  query {ids[q]}: {str(e)[:100]}")
            pick = base
        return base, pick, names[q] in cands

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(one, queries))
    n = len(res)
    print(f"\nGemini re-rank ({rerank.RERANK_MODEL}) of the combined top 5, {n} random queries:")
    print(f"  combined top-1 {sum(r[0] for r in res) / n:.3f} -> Gemini top-1 {sum(r[1] for r in res) / n:.3f}"
          f"  (ceiling: right turtle in top 5 {sum(r[2] for r in res) / n:.3f};"
          f" fixed {sum(r[1] and not r[0] for r in res)}, broke {sum(r[0] and not r[1] for r in res)})")


def sift_eval(names, ids, groups, S_emb=None):
    """SIFT on shell crops, every query against every other-sighting reference crop."""
    names, groups = np.asarray(names), np.asarray(groups)
    paths = [f"data/{shellcrop.VARIANTS['crop']['dir']}/{n}/{c}.jpg" for n, c in zip(names, ids)]
    index = sift_match.SiftIndex(names, ids, paths, identify.DATA / "sift_crop250.pkl", root=identify.ROOT)
    same_group = groups[:, None] == groups[None, :]
    queries = np.where(((names[:, None] == names[None, :]) & ~same_group).any(1))[0]
    classes = sorted(set(names))
    cidx = np.array([classes.index(n) for n in names])
    t1 = t5 = f1 = f5 = confirmed_right = confirmed_wrong = 0
    for q in queries:
        s = index.scores(index.feats[q], exclude=same_group[q])
        best = {}
        for j in np.argsort(-s):
            if np.isfinite(s[j]):
                best.setdefault(names[j], s[j])
        top = list(best)[:5]
        t1 += top[0] == names[q]
        t5 += names[q] in top
        if S_emb is not None:
            fused = matching.fuse(s, np.where(same_group[q], -np.inf, S_emb[q]), cidx, len(classes))
            ftop = [classes[k] for k in np.argsort(-fused)[:5]]
            f1 += ftop[0] == names[q]
            f5 += names[q] in ftop
        confirmed_right += best.get(names[q], 0) >= sift_match.CONFIRM
        confirmed_wrong += max(v for n, v in best.items() if n != names[q]) >= sift_match.CONFIRM
    n = len(queries)
    print(f"\nSIFT spot match (box-turtle-id's matcher) on shell crops, any view, {n} queries:")
    print(f"  top-1 {t1 / n:.3f}  top-5 {t5 / n:.3f}")
    print(f"  score >= {sift_match.CONFIRM:g}: right turtle reaches it {confirmed_right / n:.0%}, "
          f"best wrong turtle {confirmed_wrong / n:.0%}")
    if S_emb is not None:
        print(f"SIFT + combined embeddings (matching.fuse): top-1 {f1 / n:.3f}  top-5 {f5 / n:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemini", type=int, metavar="N", help="also test Gemini re-ranking on N queries")
    ap.add_argument("--sift", action="store_true", help="also test the SIFT spot matcher")
    args = ap.parse_args()
    identify.load_env()
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
    if args.gemini and ("crop", True) in sims and ("tight", True) in sims:
        gemini_eval((sims[("crop", True)] + sims[("tight", True)]) / 2, names, ids, groups, args.gemini)
    if args.sift:
        S_emb = (sims[("crop", True)] + sims[("tight", True)]) / 2 if ("crop", True) in sims else None
        sift_eval(names, ids, groups, S_emb)


if __name__ == "__main__":
    main()
