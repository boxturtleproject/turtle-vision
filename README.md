# Box-turtle individual identification

Pipeline that classifies each turtle photo by subject, embeds it with Google's
multimodal **`gemini-embedding-2-preview`** (3072 dims), and trains a k-NN
classifier to identify individual turtles from their carapace pattern.

**Prerequisites:** the dataset (`data/<turtle>/<capture_id>.jpg`) is not
committed. Download it from the live box-turtle-id app with
`python fetch_data.py` (see [Prerequisites](#prerequisites)).

## Headline result

| metric | value |
|---|---:|
| individuals (classes) | 37 (excludes `Unidentified` upstream; `Juvenile` dropped by classifier) |
| carapace photos used | 633 |
| split | 432 train / 98 valid / 98 test |
| **best test top-1** | **0.857** (PCA → 128, 1-NN cosine) |
| best test top-5 | 0.939 (PCA → 256, 5-NN cosine) |

See [`predictions_pca.csv`](data/predictions_pca.csv) for per-image predictions.

### Different-day result (`python evaluate.py`)

The split above is per photo, so a test photo can match a photo of the same
turtle taken minutes earlier, with the same background and light. That
inflates scores. `evaluate.py` scores every photo only against *other
sightings* of its turtle (534 queries; dates from `fetch_meta.py`), which is
what a new sighting looks like. Photos that box-turtle-id holds twice under
different dates (66 sets of byte-identical files, plus repeated original
filenames) are merged into one sighting, so a photo never matches its own
copy.

| matcher | top-1 | top-5 |
|---|---:|---:|
| whole photo, PCA | 0.388 | 0.684 |
| shell crop, PCA | 0.395 | 0.682 |
| tight crop, PCA | 0.440 | 0.740 |
| tight crop, LDA | 0.489 | 0.796 |
| combined: crop + tight, LDA | 0.551 | 0.839 |
| combined top 5, re-ranked by Gemini (`rerank.py`) | 0.717 | 0.839 |
| SIFT spot match, whole photo, same view (box-turtle-id as deployed) | 0.790 | 0.841 |
| SIFT spot match, whole photo, any view | 0.788 | 0.867 |
| SIFT spot match on the shell crop, any view (`sift_match.py`) | 0.815 | 0.878 |
| **SIFT + combined embeddings** (`matching.fuse`) | **0.852** | **0.925** |

Gemini choosing among the SIFT + embeddings top 5 adds little: 0.852 → 0.863
top-1 (fixed 24, broke 18), within noise for ~3.5s and an API call per photo.
With whole-photo SIFT, where the fused top-1 starts lower, it helped more
(0.837 → 0.867); choosing among SIFT's own top 5 gave 0.835. The app shows it
as a second opinion next to SIFT + embeddings.

SIFT is box-turtle-id's matcher (`backend/app/services/sift.py`), ported:
keypoints at 250px wide, ratio test 0.67, score = good matches / fewest
keypoints × 100. It compares against *every* reference photo, so unlike the
re-rankers it isn't capped by the embedding shortlist, and it doesn't need
the photo's view. Its score is also the best "new turtle" signal here: at
box-turtle-id's cut-off of 4, the best wrong turtle reaches it for ~2% of
queries and the right turtle for ~58% on shell crops (41% on whole photos,
which is why the app crops first). Adding up the evidence, each turtle
scored by log(1 + best SIFT score) + best combined-embedding similarity,
beats either alone: SIFT is decisive when spots match and the embeddings
break ties when it finds little (whole-photo SIFT fused: 0.837 / 0.925). Tried and not better: using SIFT only
to re-rank the combined top 5 (0.734), SIFT at 400px (0.710 as a re-ranker),
DISK + LightGlue keypoints with RANSAC as a re-ranker (0.736), and falling
back to Gemini when the SIFT score is low (0.800 at best).

LDA (in `matching.py`) is PCA → 128 followed by Linear Discriminant Analysis
fit on turtle names, scored out-of-fold over (turtle, day) groups. Gemini
re-ranking sends the query and 3 reference shell crops for each of the
combined matcher's top 5 turtles to `gemini-3.6-flash` and asks which is the
same individual (fixed 115 queries, broke 26; ~3.4s, ~9k input tokens each;
re-run with `python evaluate.py --gemini N`). It can't flag a new turtle:
with the true turtle removed it still picks one at ~0.98 confidence. Also tried
and not adopted: scoring a turtle by the mean of its top 2–5 photos (no
gain), favouring same-view photos (no gain), restricting to same view (worse).

"New turtle" detection by similarity barely works across days: known and new
turtles' best-match similarities overlap almost completely. At a cut-off that
still recognises 80% of known turtles, the combined matcher flags only 5% of
new ones.

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

# 7. (optional) crop references to the shell, then run the field-test app
python crop_photos.py     # ~628 Gemini box calls + embeddings
python app.py
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

Recreate it with `python fetch_data.py` (~250 MB). `capture_id` is the
`captures.id` in [box-turtle-id](https://github.com/boxturtleproject/box-turtle-id);
the script downloads each capture's 1280px display derivative — the resolution
the embeddings were computed on — from the app's public
`/api/static/captures/derivatives/display/{capture_id}.jpg` endpoint, using
`data/classifications.csv` as the manifest. Re-running skips existing files. The
folder name is the individual's identifier and is used as the class label by
the classifier.

---

## Pipeline

```
   data/<turtle_name>/<capture_id>.jpg   [~1000 imgs, fetch_data.py]
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

### `crop_photos.py` — shell crops for the reference set
For every photo in `splits.csv`: asks Gemini (`gemini-3.6-flash`, override with
`CROP_MODEL`) for the carapace bounding box (`data/crops.csv`, committed), then
crops it two ways and embeds each (full frame if no shell is found):

| variant | crop | images | vectors |
|---|---|---|---|
| `crop` | box + 5% margin | `data/crops/<turtle>/<id>.jpg` | `data/embeddings_crop.sqlite` |
| `tight` | central 71% of the box (largest rectangle inside an ellipse) — all shell, loses marginal scutes | `data/crops_tight/<turtle>/<id>.jpg` | `data/embeddings_tight.sqlite` |

Resume-safe. The crop logic lives in `shellcrop.py` and is shared with `app.py`.

### `fetch_meta.py`
Writes `data/capture_meta.csv` (committed): date and view (`carapace_top`,
`carapace_left`, `front`, `plastron`, …) for every manifest capture, from
box-turtle-id's public turtle endpoint. Used by `evaluate.py` and `app.py` to
score and calibrate on other-day photos.

### `evaluate.py`
The different-day comparison above, for every matcher. Re-run it as photos are
added. Shared maths (PCA / LDA projections, out-of-fold similarities,
cut-off calibration, ranking) lives in `matching.py`.

### `sift_match.py`
box-turtle-id's SIFT spot matcher, ported (see the results above). Reference
crop features are cached in `data/sift_crop250.pkl` (gitignored, ~70 MB).
`python evaluate.py --sift` re-runs its different-day test.

### `app.py` — field-test web app
Upload a photo (laptop, or a phone on the same Wi-Fi) and compare matchers
side by side, best first:

- **SIFT + embeddings** (best): the two added up (`matching.fuse`). Its
  banner uses the SIFT rule below.
- **Gemini pick** (second opinion): Gemini choosing among the SIFT +
  embeddings top 5, with a one-line reason; its banner re-checks the SIFT
  score of the turtle Gemini picked. Adds ~3.5s; `--no-gemini` to skip.
- **Spot match**: SIFT on the shell crop against every reference crop. A
  score of 4+ confirms a known turtle; below that the banner says it could be
  new, or a view we don't have. `--no-sift` to skip.
- **Combined** (crop + tight, LDA), **whole photo**, **cropped to shell** and
  **tight (inside shell)**: embedding matchers, each with a "weak match"
  banner whose cut-off is calibrated at startup on out-of-fold, other-day
  similarities to still recognise 80% of known turtles (`--keep-known`).

The upload is cropped on the fly with the same Gemini box prompt. Each column
shows the top-5 individuals with their nearest reference photos. Record the
true answer once (which turtle / new turtle / bad photo) and the app scores
every matcher. Uploads are EXIF-rotated, downscaled to 1280px and re-encoded
as JPEG (HEIC supported). `/crops` shows every reference crop for review.
Uploads go to `uploads/`, every upload and verdict to
`results/session_log.csv` (both gitignored).

```bash
python app.py            # prints localhost + LAN URL; --no-crop for full-frame only
```

### Gemini via Vertex AI
`embed_photos.py`, `identify.py` and `app.py` use a Gemini API key by default.
To bill through Vertex AI instead, set `GOOGLE_GENAI_USE_VERTEXAI=true`,
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in `.env` and run
`gcloud auth application-default login` (see `.env.example`).

---

## What's in `data/` (after running the pipeline)

| path | committed? | size | description |
|---|---|---:|---|
| `data/<turtle>/*.jpg` | no (gitignored) | ~330 MB | image folders, one per individual; **downloaded by `fetch_data.py`** |
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
