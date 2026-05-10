"""Embedding-based classifier for individual box-turtle ID.

Three classifiers compared, all on top of frozen 3072-d gemini-embedding-2-preview
vectors stored in data/embeddings.sqlite:

    1. Centroid (prototype): mean of L2-normalized train embeddings per class,
       re-normalized; cosine sim at inference.
    2. k-NN cosine over all train embeddings (k = 1, 3, 5).
    3. Linear probe: multinomial logistic regression with class weights.

Selection: pick best by valid top-1, then report test top-1/3/5 and per-class accuracy.
By default 'Unidentified' and 'Juvenile' are excluded as they aggregate multiple individuals.
"""
import argparse, csv, sqlite3, struct, sys, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "embeddings.sqlite"
SPLITS = DATA / "splits.csv"
PRED_OUT = DATA / "predictions.csv"
DROP_DEFAULT = ("Unidentified", "Juvenile")
DIM = 3072

def load_data(drop):
    splits = list(csv.DictReader(SPLITS.open()))
    splits = [r for r in splits if r["turtle_name"] not in drop]
    ids = [int(r["capture_id"]) for r in splits]
    con = sqlite3.connect(DB)
    rows = con.execute(
        f"SELECT capture_id, embedding FROM images "
        f"WHERE capture_id IN ({','.join('?'*len(ids))})", ids
    ).fetchall()
    emb_by_id = {cid: np.array(struct.unpack(f"<{DIM}f", blob), dtype=np.float32)
                 for cid, blob in rows}
    bundles = {"train": [], "valid": [], "test": []}
    missing = 0
    for r in splits:
        cid = int(r["capture_id"])
        if cid not in emb_by_id:
            missing += 1; continue
        bundles[r["split"]].append((cid, r["turtle_name"], r["file_path"], emb_by_id[cid]))
    if missing: print(f"WARNING: {missing} split rows had no embedding")
    return bundles

def stack(bundle):
    cids = np.array([b[0] for b in bundle])
    names = [b[1] for b in bundle]
    paths = [b[2] for b in bundle]
    X = np.stack([b[3] for b in bundle]).astype(np.float32)
    # Embeddings are returned L2-normalized by Gemini, but renormalize defensively
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return cids, names, paths, X

def build_label_index(train_names):
    classes = sorted(set(train_names))
    idx = {c: i for i, c in enumerate(classes)}
    return classes, idx

# ---------- Classifiers ----------

def fit_centroids(X, y, n_classes):
    """Mean of L2-normalized train embeddings per class, then re-normalized."""
    C = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = X[y == c]
        C[c] = m.mean(axis=0) if len(m) else 0.0
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C

def score_centroid(C, X):
    # cosine since both rows are unit-norm
    return X @ C.T

def score_knn(Xtr, ytr, X, k, n_classes):
    sim = X @ Xtr.T                            # (n_query, n_train)
    # Top-k per query
    topk_idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
    # Sum similarities per class within the top-k as the score
    scores = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    rows = np.arange(X.shape[0])[:, None]
    selected_sim = sim[rows, topk_idx]
    selected_y = ytr[topk_idx]
    np.add.at(scores, (rows.repeat(k, axis=1), selected_y), selected_sim)
    return scores

def fit_logreg(X, y):
    # n=470, d=3072, 37 classes (after dropping the two aggregate labels)
    return LogisticRegression(
        C=1.0, max_iter=2000, n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",  # multinomial by default in sklearn>=1.5
    ).fit(X, y)

# ---------- Eval ----------

def topk_acc(scores, y_true, k):
    if k > scores.shape[1]: k = scores.shape[1]
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

def per_class_acc(pred, y_true, classes):
    out = {}
    for ci, cname in enumerate(classes):
        mask = y_true == ci
        n = int(mask.sum())
        if n == 0: continue
        out[cname] = (int((pred[mask] == ci).sum()), n)
    return out

def evaluate(name, scores, y_true, classes):
    pred = scores.argmax(axis=1)
    return {
        "model": name,
        "top1": topk_acc(scores, y_true, 1),
        "top3": topk_acc(scores, y_true, 3),
        "top5": topk_acc(scores, y_true, 5),
        "per_class": per_class_acc(pred, y_true, classes),
        "pred": pred,
    }

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-aggregates", action="store_true",
                    help="Keep 'Unidentified' / 'Juvenile' classes (default: drop)")
    args = ap.parse_args()
    drop = () if args.keep_aggregates else DROP_DEFAULT

    print(f"Dropping aggregate classes: {drop or '(none)'}\n")
    bundles = load_data(drop)
    if not bundles["train"]:
        print("No training data after filtering!"); sys.exit(1)

    cids_tr, names_tr, _,    Xtr = stack(bundles["train"])
    cids_va, names_va, _,    Xva = stack(bundles["valid"])
    cids_te, names_te, paths_te, Xte = stack(bundles["test"])

    classes, idx = build_label_index(names_tr)
    n_classes = len(classes)
    ytr = np.array([idx[n] for n in names_tr])
    # Restrict valid/test to classes seen in train (should be all of them)
    keep_va = [i for i, n in enumerate(names_va) if n in idx]
    keep_te = [i for i, n in enumerate(names_te) if n in idx]
    Xva = Xva[keep_va]; names_va = [names_va[i] for i in keep_va]
    Xte = Xte[keep_te]; names_te = [names_te[i] for i in keep_te]
    cids_te = cids_te[keep_te]; paths_te = [paths_te[i] for i in keep_te]
    yva = np.array([idx[n] for n in names_va])
    yte = np.array([idx[n] for n in names_te])

    print(f"classes: {n_classes}")
    print(f"counts:  train={len(ytr)}, valid={len(yva)}, test={len(yte)}\n")

    # Fit
    centroids = fit_centroids(Xtr, ytr, n_classes)
    logreg = fit_logreg(Xtr, ytr)

    # Score on valid
    valid_scores = {
        "centroid":  score_centroid(centroids, Xva),
        "knn-k1":    score_knn(Xtr, ytr, Xva, 1, n_classes),
        "knn-k3":    score_knn(Xtr, ytr, Xva, 3, n_classes),
        "knn-k5":    score_knn(Xtr, ytr, Xva, 5, n_classes),
        "logreg":    logreg.predict_proba(Xva),
    }
    valid_results = {n: evaluate(n, s, yva, classes) for n, s in valid_scores.items()}

    print("=== VALID ===")
    print(f"  {'model':<10} {'top1':>6} {'top3':>6} {'top5':>6}")
    for n, r in valid_results.items():
        print(f"  {n:<10} {r['top1']:>6.3f} {r['top3']:>6.3f} {r['top5']:>6.3f}")
    best_name = max(valid_results, key=lambda n: valid_results[n]["top1"])
    print(f"\nBest on valid (top-1): {best_name}\n")

    # Score on test (all classifiers, but headline is the winner)
    test_scores = {
        "centroid":  score_centroid(centroids, Xte),
        "knn-k1":    score_knn(Xtr, ytr, Xte, 1, n_classes),
        "knn-k3":    score_knn(Xtr, ytr, Xte, 3, n_classes),
        "knn-k5":    score_knn(Xtr, ytr, Xte, 5, n_classes),
        "logreg":    logreg.predict_proba(Xte),
    }
    test_results = {n: evaluate(n, s, yte, classes) for n, s in test_scores.items()}

    print("=== TEST ===")
    print(f"  {'model':<10} {'top1':>6} {'top3':>6} {'top5':>6}")
    for n, r in test_results.items():
        marker = "  <-- best on valid" if n == best_name else ""
        print(f"  {n:<10} {r['top1']:>6.3f} {r['top3']:>6.3f} {r['top5']:>6.3f}{marker}")

    # Per-class for the winner
    winner = test_results[best_name]
    pca = winner["per_class"]
    print(f"\n=== Per-class TEST top-1 ({best_name}) ===")
    print(f"  {'name':<22} {'correct':>7} {'total':>5} {'acc':>5}")
    for cname in sorted(pca, key=lambda c: (-pca[c][0]/max(1,pca[c][1]), c)):
        c, n = pca[cname]
        print(f"  {cname:<22} {c:>7} {n:>5} {c/n:>5.2f}")

    # Write predictions for the winner on the test set
    winner_scores = test_scores[best_name]
    top5 = np.argsort(-winner_scores, axis=1)[:, :5]
    rows = []
    for i, cid in enumerate(cids_te):
        true_name = classes[yte[i]]
        preds = [classes[j] for j in top5[i]]
        rows.append({
            "capture_id": int(cid),
            "file_path": paths_te[i],
            "true": true_name,
            "pred_top1": preds[0],
            "correct_top1": int(preds[0] == true_name),
            "correct_top5": int(true_name in preds),
            "top5": ";".join(preds),
        })
    with PRED_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote test predictions: {PRED_OUT.relative_to(ROOT)}")
    print(f"  ({sum(r['correct_top1'] for r in rows)}/{len(rows)} top-1 correct)")

if __name__ == "__main__":
    main()
