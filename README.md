# Box-turtle individual identification

Pipeline that classifies each turtle photo by subject, embeds it with Google's
multimodal **`gemini-embedding-2-preview`** (3072 dims), and trains a k-NN
classifier to identify individual turtles from their carapace pattern.

**Prerequisites:** the dataset (`data/<turtle>/<capture_id>.jpg`) is assumed to
already exist on disk — it's provided out-of-band, not committed to the repo,
and not pulled by these scripts. See [What's in `data/`](#whats-in-data-after-running-the-pipeline)
for the expected layout.

## Headline result

| metric | value |
|---|---:|
| individuals (classes) | 37 (excludes `Unidentified` upstream; `Juvenile` dropped by classifier) |
| carapace photos used | 633 |
| split | 432 train / 98 valid / 98 test |
| **best test top-1** | **0.857** (PCA → 128, 1-NN cosine) |
| best test top-5 | 0.939 (PCA → 256, 5-NN cosine) |

See [`predictions_pca.csv`](data/predictions_pca.csv) for per-image predictions.

---

## Quick start

Assumes `data/<turtle>/<capture_id>.jpg` is already in place (see
[Prerequisites](#prerequisites)).

```bash
# 1. clone, create venv, install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. set up API keys (only needed for steps 3 and 4)
cp .env.example .env
$EDITOR .env                       # paste GEMINI_API_KEY + ANTHROPIC_API_KEY

# 3. (optional) re-classify images — needs ANTHROPIC_API_KEY
#    classifications.csv is committed; only re-run if you change the prompt
python classify_photos.py

# 4. embed every image — needs GEMINI_API_KEY (~5 min, ~$0.20)
python embed_photos.py

# 5. build train/valid/test split over carapace photos
python make_splits.py

# 6. train + evaluate the classifier (no API calls; ~10 sec)
python classify.py        # baseline: prototype, k-NN, logistic regression
python classify_pca.py    # PCA + 1-NN sweep (recommended)
```

`data/classifications.csv` and `data/splits.csv` are committed, so you can skip
straight to step 5 if you don't want to re-classify. Step 4 (embeddings) is
**required** for the classifier — the SQLite is excluded from git, so a
`GEMINI_API_KEY` is needed to regenerate it.

### Prerequisites

These scripts assume the image dataset is already on disk in this layout:

```
data/
├── Aztec/         # one folder per individual turtle
│   ├── 188.jpg    # filenames are integer capture IDs
│   ├── 189.jpg
│   └── ...
├── Shingles/
└── ...            # 39 folders total in the reference dataset
```

The dataset is provided separately (it's ~330 MB and not committed). The
folder name is the individual's identifier and is used as the class label by
the classifier.

---

## Pipeline

```
   data/<turtle_name>/<capture_id>.jpg   [~1000 imgs, provided out-of-band]
                        │
                        ▼
   classify_photos.py  ──►  data/classifications.csv              [committed]
       (Claude Haiku 4.5 vision; carapace / plastron / habitat /
        other_closeup × photo / illustration)
                        │
                        ▼
   embed_photos.py     ──►  data/embeddings.sqlite                [13 MB, gitignored]
       (gemini-embedding-2-preview, 3072 dims, float32 BLOB)
                        │
                        ▼
   make_splits.py      ──►  data/splits.csv                       [committed]
       (exclude 'Unidentified' folder; filter to carapace photos;
        ~70/15/15; min ≥2 train + 1 valid + 1 test)
                        │
                        ▼
   classify.py / classify_pca.py
                        ──►  data/predictions.csv  data/predictions_pca.csv
       (k-NN cosine on the embeddings; PCA optional)
```

---

## Scripts

### `classify_photos.py`
Uses **Claude Haiku 4.5** (`anthropic` SDK, structured output via Pydantic) to
tag each image on two independent axes:

- **category**: `carapace` · `plastron` · `other_closeup` · `habitat`
- **media_type**: `photo` · `illustration`

Output appended to `data/classifications.csv`. Resume-safe: re-running only
processes images that aren't already in the CSV.

`test_classify.py` — sanity check on 4 reference images with known labels.

### `embed_photos.py`
For every `data/*/*.jpg`, computes a 3072-dim embedding with
`gemini-embedding-2-preview` (`google-genai` SDK) and stores it as a float32
BLOB in `data/embeddings.sqlite`. Resume-safe: only embeds rows missing an
entry for the configured `(model, dim)` combo.

```sql
images(
  capture_id   INTEGER PRIMARY KEY,
  turtle_name  TEXT,            -- folder name, e.g. 'Shingles'
  file_path    TEXT UNIQUE,     -- relative to repo root
  bytes        INTEGER,
  sha1         TEXT,            -- of the image bytes
  model        TEXT,            -- 'gemini-embedding-2-preview'
  dim          INTEGER,         -- 3072
  embedded_at  TEXT,
  embedding    BLOB             -- float32 LE, 12288 bytes (3072 × 4)
)
```

Read a vector:
```python
import struct, sqlite3
con = sqlite3.connect("data/embeddings.sqlite")
blob = con.execute("SELECT embedding FROM images WHERE capture_id=?", (188,)).fetchone()[0]
v = struct.unpack("<3072f", blob)   # tuple of 3072 floats, L2-normalized
```

### `make_splits.py`
Excludes the `Unidentified` folder (it aggregates many individuals), filters
`classifications.csv` to **carapace photos only** (drops illustrations and
non-carapace categories), keeps individuals with ≥4 carapace photos, then
splits ~70/15/15 with hard minimums (≥2 train, ≥1 valid, ≥1 test). Seeded
(`SEED=42`) for reproducibility. Writes `data/splits.csv`.

### `classify.py`
Compares three classifiers on the frozen embeddings and reports valid + test
top-1/3/5 accuracy plus per-class breakdown:

1. **Centroid** — mean of L2-normalized train embeddings per class.
2. **k-NN cosine** (k=1, 3, 5) over all train embeddings.
3. **Logistic regression** linear probe.

By default it also drops `Juvenile` (5 photos, multi-individual bin); pass
`--keep-aggregates` to include it. (`Unidentified` is already excluded
upstream by `make_splits.py`, so it never reaches the classifier.)

Writes `data/predictions.csv` for the model that won on `valid`.

### `classify_pca.py`
Sweeps PCA dims `{32, 64, 128, 256, 512, 1024, 3072}` × whiten `{False, True}`
× k `{1, 3, 5}` for 1-NN-style cosine ranking, picks the best on `valid`,
reports test top-1/3/5 + per-class. Writes `data/predictions_pca.csv`. Also
drops `Juvenile` (and inherits the upstream `Unidentified` exclusion).
Whitening currently never wins — it loses 5–10pp at medium dims and
collapses to ~0.2 top-1 near full rank, where it amplifies noise from
small-eigenvalue components.

### `visualize.py`
Computes 2-D **t-SNE** and **UMAP** projections (cosine metric) of every
carapace photo embedding (633 photos, 38 individuals; `Unidentified`
excluded). Writes a single self-contained HTML — `embedding_viz.html` at
the repo root — with a full-screen scatter plot, t-SNE / UMAP toggle
(default t-SNE), color-coded by individual, image preview on hover, and a
click-to-isolate legend.

```bash
python visualize.py
open embedding_viz.html
```

Requires `umap-learn`. No API calls — operates on the cached embeddings in
`data/embeddings.sqlite`. The HTML is gitignored: it references the
gitignored image folders by relative path (`data/<turtle>/<id>.jpg`).

### `identify.py`
One-shot single-image classifier. Embeds a new photo with
`gemini-embedding-2-preview`, then runs the headline classifier
(PCA→128, no-whiten, 1-NN cosine) using **all 628 labeled images
(train + valid + test, minus `Juvenile`)** as the reference pool.

```bash
python identify.py path/to/photo.jpg              # default: PCA=128, k=1
python identify.py path/to/photo.jpg --top 5 --k 5
```

Prints the top-N candidate individuals and the nearest reference images
(with cosine similarity). Needs `GEMINI_API_KEY` and an existing
`data/embeddings.sqlite`.

---

## What's in `data/` (after running the pipeline)

| path | committed? | size | description |
|---|---|---:|---|
| `data/<turtle>/*.jpg` | no (gitignored) | ~330 MB | image folders, one per individual; **provided out-of-band** |
| `data/classifications.csv` | **yes** | ~55 KB | `capture_id, turtle_name, file_path, category, media_type, confidence` |
| `data/splits.csv` | **yes** | ~46 KB | `… , split` (`train`/`valid`/`test`) over carapace photos |
| `data/predictions.csv` | **yes** | ~5 KB | test-set predictions from `classify.py` (winner on valid) |
| `data/predictions_pca.csv` | **yes** | ~5 KB | test-set predictions from `classify_pca.py` |
| `data/embeddings.sqlite` | no (gitignored) | ~13 MB | 1000 × float32 × 3072 vectors + metadata |

Image folders and the SQLite are gitignored to keep the repo small. The
SQLite is regenerated locally with `embed_photos.py` (needs `GEMINI_API_KEY`).
Keys live in `.env` (gitignored — see `.env.example`).

---

## Notes / caveats

- **`Unidentified` (49 carapace photos)** and **`Juvenile` (5)** are bin labels
  in the source data, not single individuals. `Unidentified` is excluded by
  `make_splits.py` so it never appears in train/valid/test. `Juvenile` is also
  dropped by default in `classify*.py` — pass `--keep-aggregates` to include
  it (e.g. for open-set / unknown-detection experiments).
- Classes are heavily imbalanced (Cruella has 52 train images, Emoji has 2).
  k-NN handles this well; logistic regression overfits.
- Embeddings are computed on the **whole frame**, not on a cropped carapace.
  Tight crops would likely improve accuracy meaningfully — a useful next step.
- The provided JPEGs are ~280 KB display-resolution derivatives. Higher-res
  originals were not available.
- API costs are roughly: ~$0.20 for 1000 Gemini embeddings, ~$0.30 for 1000
  Haiku classifications. Both APIs are resume-safe so partial re-runs are cheap.

---

## Reproduce results

```bash
source .venv/bin/activate
python make_splits.py    # uses committed classifications.csv (no API keys)
python classify_pca.py   # needs data/embeddings.sqlite — re-run embed_photos.py if missing
```

The classifier is deterministic for a given embedding set + `SEED=42`.
