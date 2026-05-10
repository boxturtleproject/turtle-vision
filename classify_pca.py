"""PCA + 1-NN sweep on top of the frozen Gemini embeddings.

For each target dim d in {32, 64, 128, 256, 512, 1024, 3072}:
  - Fit PCA on train (no whitening)
  - L2-normalize projected train/valid/test
  - 1-NN cosine + 3-NN, 5-NN
Pick the (model, k) with the best valid top-1; report TEST top-1/3/5 and per-class
accuracy. 'Unidentified' / 'Juvenile' dropped (aggregate labels).
"""
import csv, sqlite3, struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "embeddings.sqlite"
SPLITS = DATA / "splits.csv"
PRED_OUT = DATA / "predictions_pca.csv"
DROP = ("Unidentified", "Juvenile")
DIM = 3072
PCA_DIMS = [32, 64, 128, 256, 512, 1024, 3072]
KS = [1, 3, 5]

def load():
    splits = list(csv.DictReader(SPLITS.open()))
    splits = [r for r in splits if r["turtle_name"] not in DROP]
    ids = [int(r["capture_id"]) for r in splits]
    con = sqlite3.connect(DB)
    rows = con.execute(
        f"SELECT capture_id, embedding FROM images "
        f"WHERE capture_id IN ({','.join('?'*len(ids))})", ids
    ).fetchall()
    emb = {cid: np.array(struct.unpack(f"<{DIM}f", b), dtype=np.float32) for cid, b in rows}
    bundles = {"train": [], "valid": [], "test": []}
    for r in splits:
        cid = int(r["capture_id"])
        if cid in emb:
            bundles[r["split"]].append((cid, r["turtle_name"], r["file_path"], emb[cid]))
    return bundles

def stack(bundle):
    X = np.stack([b[3] for b in bundle]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (
        np.array([b[0] for b in bundle]),
        [b[1] for b in bundle],
        [b[2] for b in bundle],
        X,
    )

def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def knn_scores(Xtr, ytr, X, k, n_classes):
    sim = X @ Xtr.T
    k_eff = min(k, sim.shape[1])
    topk = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    scores = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    rows = np.arange(X.shape[0])[:, None]
    np.add.at(scores, (rows.repeat(k_eff, axis=1), ytr[topk]), sim[rows, topk])
    return scores

def topk_acc(scores, y, k):
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))

def main():
    b = load()
    cids_tr, names_tr, _,        Xtr = stack(b["train"])
    cids_va, names_va, _,        Xva = stack(b["valid"])
    cids_te, names_te, paths_te, Xte = stack(b["test"])
    classes = sorted(set(names_tr))
    idx = {c: i for i, c in enumerate(classes)}
    ytr = np.array([idx[n] for n in names_tr])
    yva = np.array([idx[n] for n in names_va])
    yte = np.array([idx[n] for n in names_te])
    n_classes = len(classes)
    print(f"classes={n_classes} train={len(ytr)} valid={len(yva)} test={len(yte)}\n")

    rows_v, rows_t = [], []
    cache = {}  # (dim) -> (Xtr_p, Xva_p, Xte_p)

    max_pca = min(Xtr.shape[0], Xtr.shape[1])  # PCA can't exceed n_samples
    dims = sorted({d if d <= max_pca else max_pca for d in PCA_DIMS})

    print(f"=== VALID top-1 sweep (max PCA dim = {max_pca}) ===")
    print(f"  {'dim':>4} | {'k=1':>5} {'k=3':>5} {'k=5':>5}")
    for d in dims:
        if d >= Xtr.shape[1]:
            Xtr_p, Xva_p, Xte_p = Xtr, Xva, Xte
        else:
            pca = PCA(n_components=d, whiten=False, svd_solver="full", random_state=0)
            Xtr_p = pca.fit_transform(Xtr)
            Xva_p = pca.transform(Xva)
            Xte_p = pca.transform(Xte)
        Xtr_p = l2norm(Xtr_p); Xva_p = l2norm(Xva_p); Xte_p = l2norm(Xte_p)
        cache[d] = (Xtr_p, Xva_p, Xte_p)
        line_v = [f"{d:>4} |"]
        for k in KS:
            s = knn_scores(Xtr_p, ytr, Xva_p, k, n_classes)
            a = topk_acc(s, yva, 1)
            line_v.append(f"{a:>5.3f}")
            rows_v.append((d, k, "valid", a, topk_acc(s, yva, 3), topk_acc(s, yva, 5)))
        print("  " + " ".join(line_v))

    # Pick best by valid top-1 (tiebreak: lower dim, then lower k)
    best = max(rows_v, key=lambda r: (r[3], -r[0], -r[1]))
    best_d, best_k = best[0], best[1]
    print(f"\nBest on valid: dim={best_d}, k={best_k}  (valid top-1={best[3]:.3f}, top-3={best[4]:.3f}, top-5={best[5]:.3f})\n")

    # Full TEST table
    print(f"=== TEST ===")
    print(f"  {'dim':>4} {'k':>3} | {'top1':>5} {'top3':>5} {'top5':>5}")
    test_table = []
    for d in dims:
        Xtr_p, _, Xte_p = cache[d]
        for k in KS:
            s = knn_scores(Xtr_p, ytr, Xte_p, k, n_classes)
            t1, t3, t5 = topk_acc(s, yte, 1), topk_acc(s, yte, 3), topk_acc(s, yte, 5)
            mark = "  <-- best on valid" if (d == best_d and k == best_k) else ""
            print(f"  {d:>4} {k:>3} | {t1:>5.3f} {t3:>5.3f} {t5:>5.3f}{mark}")
            test_table.append((d, k, t1, t3, t5, s))

    # Per-class for the winner on test
    s_win = next(r[5] for r in test_table if r[0] == best_d and r[1] == best_k)
    pred = s_win.argmax(axis=1)
    per_class = defaultdict(lambda: [0, 0])
    for i, t in enumerate(yte):
        per_class[classes[t]][1] += 1
        if pred[i] == t: per_class[classes[t]][0] += 1
    print(f"\n=== Per-class TEST top-1 (dim={best_d}, k={best_k}) ===")
    print(f"  {'name':<22} {'correct':>7} {'total':>5} {'acc':>5}")
    for cname in sorted(per_class, key=lambda c: (-per_class[c][0]/max(1, per_class[c][1]), c)):
        c, n = per_class[cname]
        print(f"  {cname:<22} {c:>7} {n:>5} {c/n:>5.2f}")

    # Write predictions for the winner
    top5_idx = np.argsort(-s_win, axis=1)[:, :5]
    out_rows = []
    for i, cid in enumerate(cids_te):
        true = classes[yte[i]]
        preds = [classes[j] for j in top5_idx[i]]
        out_rows.append({
            "capture_id": int(cid),
            "file_path": paths_te[i],
            "true": true,
            "pred_top1": preds[0],
            "correct_top1": int(preds[0] == true),
            "correct_top5": int(true in preds),
            "top5": ";".join(preds),
            "config": f"pca{best_d}_k{best_k}",
        })
    with PRED_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nWrote test predictions: {PRED_OUT.relative_to(ROOT)}")
    print(f"  ({sum(r['correct_top1'] for r in out_rows)}/{len(out_rows)} top-1 correct)")

if __name__ == "__main__":
    main()
