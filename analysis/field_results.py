"""Compare the matchers on a verified field session.

Reads a session's CSVs (results/ by default, or a snapshot such as
analysis/field_2026-10/) after every photo has an answer recorded on the
review page, and prints:

  1. known turtles: right first time / in top 5, per method
  2. spot-score percentiles for known, new and bad photos
  3. spot-score bands: what the photos in each band turned out to be
  4. the "new turtle" rule: spot >= T names the pick, else "new turtle"
  5. how well each score separates known from new (AUC)
  6. known-turtle misses, per turtle
  7. encounters: photos of one known turtle on one day, combined

    python analysis/field_results.py analysis/field_2026-10
"""
import collections
import csv
import sys
from pathlib import Path

import numpy as np

METHODS = [("best", "SIFT + embeddings"), ("sift", "SIFT only"), ("combined", "Embeddings only (crop+tight)"),
           ("tight", "Embeddings, tight crop"), ("crop", "Embeddings, shell crop"),
           ("full", "Embeddings, whole photo"), ("gemini", "Gemini pick")]
BANDS = [("Confirmed 4+", 4, 1e9), ("Likely 2-4", 2, 4), ("Possible 1-2", 1, 2), ("No match <1", -1, 1)]


def load(folder: Path):
    rows = list(csv.DictReader((folder / "uploads_summary.csv").open()))
    log = list(csv.DictReader((folder / "session_log.csv").open()))
    taken = {}
    if (folder / "photo_meta.csv").exists():
        taken = {r["upload_id"]: r["taken"] for r in csv.DictReader((folder / "photo_meta.csv").open())}
    pred = {(r["upload_id"], r["method"]): r for r in log if r["event"] == "identify"}
    for r in rows:
        a = r["true_answer"]
        r["kind"] = "bad" if a == "bad photo" else "new" if a == "new turtle" else "" if not a else "known"
        r["spot"] = float(r["best_spot_score"]) if r["best_spot_score"] else None
        r["day"] = (taken.get(r["upload_id"]) or r["time"])[:10]
    return rows, pred


def top5(pred, uid, m):
    return [x.rsplit(":", 1)[0] for x in pred[(uid, m)]["top5"].split(";")]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def pctl(a):
    return np.percentile(a, [10, 25, 50, 75, 90]).round(2).tolist() if len(a) else []


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows, pred = load(folder)
    known = [r for r in rows if r["kind"] == "known" and r["spot"] is not None]
    new = [r for r in rows if r["kind"] == "new" and r["spot"] is not None]
    bad = [r for r in rows if r["kind"] == "bad" and r["spot"] is not None]
    print(f"{len(rows)} photos: {len(known)} known, {len(new)} new, {len(bad)} bad, "
          f"{sum(not r['kind'] for r in rows)} without an answer")

    print("\n1. Known turtles: right first time / right turtle in top 5")
    for m, label in METHODS:
        have = [r for r in known if (r["upload_id"], m) in pred]
        if have:
            t1 = sum(top5(pred, r["upload_id"], m)[0] == r["true_answer"] for r in have)
            t5 = sum(r["true_answer"] in top5(pred, r["upload_id"], m) for r in have)
            print(f"  {label:30s} {t1:3d}/{len(have)} = {t1 / len(have):5.1%}   top 5 {t5 / len(have):5.1%}")

    print("\n2. Spot-score percentiles (10/25/50/75/90)")
    for name, group in (("known", known), ("new", new), ("bad", bad)):
        print(f"  {name:5s} {pctl([r['spot'] for r in group])}")

    print("\n3. Bands")
    for name, lo, hi in BANDS:
        K = [r for r in known if lo <= r["spot"] < hi]
        N = [r for r in new if lo <= r["spot"] < hi]
        B = [r for r in bad if lo <= r["spot"] < hi]
        kr = sum(r["best_pick"] == r["true_answer"] for r in K)
        print(f"  {name:13s} {len(K) + len(N) + len(B):3d} photos | known {len(K):3d} (pick right {kr:3d})"
              f" | new {len(N):3d} | bad {len(B):3d}")

    print("\n4. Rule: spot >= T names the best pick, otherwise 'new turtle' (known + new photos)")
    print("     T  known named right  known named wrong  known called new |  new caught  new named  overall right")
    for T in (0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5):
        named = [r for r in known if r["spot"] >= T]
        kr = sum(r["best_pick"] == r["true_answer"] for r in named)
        kw, kc = len(named) - kr, len(known) - len(named)
        nc = sum(r["spot"] < T for r in new)
        print(f"  {T:4}  {kr / len(known):15.0%}  {kw / len(known):17.0%}  {kc / len(known):16.0%} |"
              f"  {nc / len(new):10.0%}  {1 - nc / len(new):9.0%}  {(kr + nc) / (len(known) + len(new)):13.0%}")

    print("\n5. Separating known from new (AUC: 0.5 = coin flip, 1 = perfect)")
    print(f"  {'SIFT + embeddings spot score':30s} {auc([r['spot'] for r in known], [r['spot'] for r in new]):.3f}")
    for m, label in (("sift", "SIFT score"), ("combined", "Embeddings-only similarity"), ("full", "Whole-photo similarity")):
        if all((r["upload_id"], m) in pred for r in known + new):
            s = lambda r: float(pred[(r["upload_id"], m)]["top1_sim"])
            print(f"  {label:30s} {auc([s(r) for r in known], [s(r) for r in new]):.3f}")

    print("\n6. Known-turtle misses (SIFT + embeddings)")
    wrong = [r for r in known if r["best_pick"] != r["true_answer"]]
    print(f"  {len(wrong)} wrong; right turtle in top 5 for {sum(r['true_answer'] in r['best_top5'].split('; ') for r in wrong)}")
    per = collections.defaultdict(lambda: [0, 0])
    for r in known:
        per[r["true_answer"]][0] += r["best_pick"] == r["true_answer"]
        per[r["true_answer"]][1] += 1
    print("  per turtle:", ", ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(per.items())))

    print("\n7. Encounters (one known turtle, one day): combined by adding each turtle's score across photos")
    enc = collections.defaultdict(list)
    for r in known:
        enc[(r["true_answer"], r["day"])].append(r)
    multi = {k: v for k, v in enc.items() if len(v) > 1}
    print(f"  {len(known)} photos in {len(enc)} encounters ({len(multi)} with 2+ photos)")

    def combined_top(rs, m):
        tot = collections.Counter()
        parts = []
        for r in rs:
            d = {x.rsplit(":", 1)[0]: float(x.rsplit(":", 1)[1]) for x in pred[(r["upload_id"], m)]["top5"].split(";")}
            parts.append((d, min(d.values())))  # turtles outside a photo's top 5 get its 5th-place score
        for n in set().union(*(d for d, _ in parts)):
            tot[n] = sum(d.get(n, floor) for d, floor in parts)
        return [n for n, _ in tot.most_common(5)]

    for m, label in METHODS[:3]:
        single = np.mean([top5(pred, r["upload_id"], m)[0] == r["true_answer"] for r in known])
        per_enc = np.mean([combined_top(v, m)[0] == t for (t, _), v in enc.items()])
        multi_single = np.mean([np.mean([top5(pred, r["upload_id"], m)[0] == t for r in v]) for (t, _), v in multi.items()])
        multi_comb = np.mean([combined_top(v, m)[0] == t for (t, _), v in multi.items()])
        print(f"  {label:30s} single photo {single:5.1%} | per encounter {per_enc:5.1%} | "
              f"2+ photo encounters: {multi_single:5.1%} -> {multi_comb:5.1%}")
    for T in (1, 1.5, 2):
        print(f"  cut-off {T}: known called new {np.mean([r['spot'] < T for r in known]):.0%} per photo -> "
              f"{np.mean([max(r['spot'] for r in v) < T for v in enc.values()]):.0%} per encounter (best photo)")


if __name__ == "__main__":
    main()
