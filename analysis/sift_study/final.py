"""Full-set comparison of the finalists (run lab_sift.py '<cfg>' full for {}, {"root": true}
and {"root": true, "pre": "lab_b"} first), with photos fixed/broken vs today's matcher and
photos the embeddings rescue/break.

    python final.py
"""
import json
import numpy as np
from lab_eval import load, evaluate, combine, score_vec, per_class, emb_class, fuse, COLS, CLASSES, CIDX, NAMES, GROUPS, FLD, REF

G = load({}, full=True)
R = load({"root": True}, full=True)
RY = load({"root": True, "pre": "lab_b"}, full=True)
ent = [("today: grey, ratio 0.67", G, "base"),
       ("grey + mutual 0.8", G, "mutn@0.8"),
       ("grey + mutual 0.8 (sqrt norm)", G, "mutg@0.8"),
       ("RootSIFT + mutual 0.8", R, "mutn@0.8"),
       ("RootSIFT + mutual 0.8 (sqrt)", R, "mutg@0.8"),
       ("Root+yellow + mutual 0.8", RY, "mutn@0.8"),
       ("Root+yellow + mutual 0.8 (sqrt)", RY, "mutg@0.8"),
       ("Root+yellow LNBNN", RY, "lnbnn_ref"),
       ("Root+yellow mutual+position", RY, "rs_mutaff")]
for k in ("mutn@0.8", "mutg@0.8"):
    C, tag = combine(R, RY, k)
    ent.append((f"Root grey+yellow summed ({k[:4]})", C, tag))
print(f"{'version':36s} " + " ".join(f"{c:>7s}" for c in COLS))
res = {}
for label, S, kind in ent:
    o = evaluate(S, kind)
    res[label] = o
    print(f"{label:36s} " + " ".join(f"{o[c]:7.3f}" for c in COLS), flush=True)
print("\nEmbeddings: photos rescued (SIFT wrong -> combined right) / broken (SIFT right -> combined wrong)")
for label, o in res.items():
    for part in ("fld", "ref"):
        a, b = o[f"_{part}_sift_ok"], o[f"_{part}_fused_ok"]
        print(f"  {label:36s} {part}: rescued {int((~a & b).sum()):3d}  broken {int((a & ~b).sum()):3d}  of {len(a)}")
base = res["today: grey, ratio 0.67"]
print("\nVs today (SIFT + embeddings, right first time): photos fixed / broken")
for label, o in res.items():
    for part in ("fld", "ref"):
        a, b = base[f"_{part}_fused_ok"], o[f"_{part}_fused_ok"]
        print(f"  {label:36s} {part}: fixed {int((~a & b).sum()):3d}  broken {int((a & ~b).sum()):3d}")
json.dump({k: {c: float(v) for c, v in o.items() if not c.startswith("_")} for k, o in res.items()}, open("final_results.json", "w"), indent=1)
