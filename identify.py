"""Identify a single turtle photo against the full labeled set.

Embeds the query image with gemini-embedding-2-preview, then runs the headline
classifier (PCA -> 128, no-whiten, 1-NN cosine) using train+valid+test as the
reference pool. 'Unidentified' is already excluded upstream by make_splits.py;
'Juvenile' is dropped here.

Usage:
    python identify.py path/to/photo.jpg [--top 5] [--pca 128] [--k 1] [--whiten]
"""
import argparse, csv, mimetypes, os, sqlite3, struct, sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "embeddings.sqlite"
SPLITS = DATA / "splits.csv"
ENV = ROOT / ".env"
MODEL = "gemini-embedding-2-preview"
DIM = 3072
DROP = ("Juvenile",)  # 'Unidentified' already excluded by make_splits.py

# Defaults: the headline winner from the PCA sweep (best test top-1 = 0.857).
DEFAULT_PCA = 128
DEFAULT_K = 1
DEFAULT_WHITEN = False


def load_env():
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def load_reference(drop):
    """Return (names, paths, X) for every labeled image we have an embedding for."""
    splits = list(csv.DictReader(SPLITS.open()))
    splits = [r for r in splits if r["turtle_name"] not in drop]
    ids = [int(r["capture_id"]) for r in splits]
    con = sqlite3.connect(DB)
    rows = con.execute(
        f"SELECT capture_id, embedding FROM images "
        f"WHERE capture_id IN ({','.join('?'*len(ids))})", ids
    ).fetchall()
    emb = {cid: np.array(struct.unpack(f"<{DIM}f", b), dtype=np.float32)
           for cid, b in rows}
    names, paths, vecs = [], [], []
    for r in splits:
        cid = int(r["capture_id"])
        if cid not in emb:
            continue
        names.append(r["turtle_name"])
        paths.append(r["file_path"])
        vecs.append(emb[cid])
    X = np.stack(vecs).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return names, paths, X


def embed_image(path: Path) -> np.ndarray:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    raw = path.read_bytes()
    mt = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    result = client.models.embed_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=raw, mime_type=mt)],
        config=types.EmbedContentConfig(output_dimensionality=DIM),
    )
    v = np.array(result.embeddings[0].values, dtype=np.float32)
    if v.shape[0] != DIM:
        raise ValueError(f"unexpected embedding dim {v.shape[0]}")
    return v / (np.linalg.norm(v) + 1e-12)


def identify(image_path: Path, top: int, pca_dim: int, k: int, whiten: bool):
    names, paths, Xref = load_reference(DROP)
    classes = sorted(set(names))
    name_to_class = {n: i for i, n in enumerate(classes)}
    yref = np.array([name_to_class[n] for n in names])

    print(f"reference: {len(names)} images, {len(classes)} individuals", file=sys.stderr)
    print(f"config:    PCA={pca_dim}, whiten={whiten}, k={k}", file=sys.stderr)
    print(f"embedding query image…", file=sys.stderr)

    q = embed_image(image_path)  # (3072,)

    # Fit PCA on the full reference set; project ref + query together.
    eff_dim = min(pca_dim, Xref.shape[0], Xref.shape[1])
    if eff_dim < Xref.shape[1]:
        pca = PCA(n_components=eff_dim, whiten=whiten,
                  svd_solver="full", random_state=0)
        Xref_p = pca.fit_transform(Xref)
        q_p = pca.transform(q[None, :])[0]
    else:
        Xref_p, q_p = Xref, q

    # L2-normalize so dot product == cosine similarity.
    Xref_p = Xref_p / (np.linalg.norm(Xref_p, axis=1, keepdims=True) + 1e-12)
    q_p = q_p / (np.linalg.norm(q_p) + 1e-12)

    sim = Xref_p @ q_p                              # (n_ref,)
    k_eff = min(k, sim.shape[0])
    topk_idx = np.argpartition(-sim, k_eff - 1)[:k_eff]
    topk_idx = topk_idx[np.argsort(-sim[topk_idx])]  # sort hits by sim desc

    # Aggregate top-k votes by class (sum of similarities).
    scores = np.zeros(len(classes), dtype=np.float32)
    for j in topk_idx:
        scores[yref[j]] += sim[j]
    order = np.argsort(-scores)

    # Print top-N classes.
    print(f"\nTop {top} predictions:")
    print(f"  {'rank':>4}  {'individual':<20} {'score':>7}")
    shown = 0
    for ci in order:
        if scores[ci] <= 0:
            break
        print(f"  {shown+1:>4}  {classes[ci]:<20} {scores[ci]:>7.3f}")
        shown += 1
        if shown >= top:
            break

    # Show the actual nearest neighbors used for the decision.
    print(f"\nNearest reference images (k={k_eff}):")
    print(f"  {'sim':>6}  {'individual':<20}  file")
    for j in topk_idx:
        print(f"  {sim[j]:>6.3f}  {names[j]:<20}  {paths[j]}")

    return classes[int(order[0])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="path to a JPEG/PNG of a turtle")
    ap.add_argument("--top", type=int, default=5, help="number of class predictions to show")
    ap.add_argument("--pca", type=int, default=DEFAULT_PCA, help="PCA target dim (default: 128)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="k for k-NN (default: 1)")
    ap.add_argument("--whiten", action="store_true", default=DEFAULT_WHITEN,
                    help="enable PCA whitening (default: off)")
    args = ap.parse_args()

    if not args.image.exists():
        ap.error(f"image not found: {args.image}")
    if not DB.exists():
        ap.error(f"missing {DB} — run embed_photos.py first")
    if not SPLITS.exists():
        ap.error(f"missing {SPLITS} — run make_splits.py first")

    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        ap.error("GEMINI_API_KEY not set (see .env)")

    pred = identify(args.image, args.top, args.pca, args.k, args.whiten)
    print(f"\nPredicted: {pred}")


if __name__ == "__main__":
    main()
