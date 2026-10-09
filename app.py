"""Local web app for field-testing turtle identification.

Upload a carapace photo (from a laptop or a phone on the same Wi-Fi) and see
the closest known individuals from several matchers side by side:

  best     — SIFT + combined embeddings: each turtle scored by
             log(1 + best SIFT score) + best embedding similarity
             (matching.fuse). Best on the different-day test. Its banner uses
             the SIFT rule: the top turtle's spot-match score >= 4 confirms it.
  sift     — box-turtle-id's SIFT spot matcher on the shell crop against every
             reference crop (sift_match.py). ~82% top-1 on the different-day
             test. Score >= 4 confirms a known turtle; below that the banner
             says it could be new.
  gemini   — second opinion: Gemini choosing among the best-guess top 5 by
             comparing the shell photos (rerank.py), with a one-line reason.
             On the different-day test it barely helps (0.852 -> 0.863 top-1;
             fixed 24, broke 18). Falls back to the combined top 5 without
             SIFT. Adds ~3.5s per upload; --no-gemini to skip.
  combined — crop + tight embeddings, each projected with LDA (learned from
             turtle names), similarities averaged. Best embedding matcher on
             the different-day test (evaluate.py): ~55% top-1, ~84% top-5.
  full     — embed the whole photo, PCA -> 128
  crop     — crop to the shell with a Gemini bounding box (+5% margin), PCA
  tight    — same box, keep only its central 71% (all shell), PCA

All rank turtles by their single most similar reference photo. You record the
true answer once (which turtle / new turtle / bad photo) and the app scores
every matcher against it. Every upload and verdict is appended to
results/session_log.csv.

A "weak match" cut-off is calibrated per matcher at startup on out-of-fold,
other-day similarities (matching.calibrate). Across days, known and new
turtles overlap heavily, so by default the cut-off is set to still recognise
80% of known turtles (--keep-known); treat the banner as a hint.

    python app.py                 # http://localhost:8000 (+ LAN URL printed)
    python app.py --no-crop       # whole-photo matcher only
    python app.py --keep-known 0.9
    python app.py --no-gemini     # skip the Gemini re-rank column
    python app.py --no-sift       # skip the SIFT spot-match column
"""
import argparse, csv, html, io, socket, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

import identify
import matching
import rerank
import shellcrop
import sift_match

register_heif_opener()  # iPhone HEIC uploads

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
LOG = ROOT / "results" / "session_log.csv"
LOG_FIELDS = ["time", "upload_id", "event", "method", "filename", "box", "top1",
              "top1_sim", "top5", "likely_new", "verdict", "true_name", "notes"]
VERDICTS = ("known", "new_turtle", "bad_photo")
MAX_SIDE = 1280  # match the reference photos (box-turtle-id display derivatives)


class Index:
    """One embedding source (full / crop / tight) projected with PCA or PCA->LDA."""

    def __init__(self, source: str, db: Path, pca_dim: int, lda: bool = False, keep_known=None):
        self.source, self.display = source, source
        self.names, paths, X = identify.load_reference(identify.DROP, db)
        self.ids = [matching.capture_id_of(p) for p in paths]
        ref_dir = shellcrop.VARIANTS.get(source, {}).get("dir")
        # show the matching reference crops, e.g. data/crops_tight/<turtle>/<id>.jpg
        self.paths = [p.replace("data/", f"data/{ref_dir}/", 1) for p in paths] if ref_dir else paths
        self.classes = sorted(set(self.names))
        self.groups = matching.turtle_day_groups(self.ids, self.names)
        self.project = matching.fit_projection(X, np.array(self.names), pca_dim, lda)
        self.X = self.project(X)
        self.S = matching.oof_sims(X, self.names, self.groups, pca_dim, lda)
        self.threshold, self.calib = matching.calibrate(self.S, self.names, self.groups, keep_known)

    def sims(self, vecs):
        return self.X @ self.project(vecs[self.source][None, :])[0]

    def query(self, vecs):
        return matching.rank(self.sims(vecs), self.names, self.paths)


class Combined:
    """Average of several Index similarities over the same reference photos."""

    def __init__(self, parts: list[Index], keep_known=None):
        assert all(p.ids == parts[0].ids for p in parts), "embedding DBs cover different photos"
        self.parts, self.display = parts, parts[-1].display
        self.names, self.paths, self.classes = parts[0].names, parts[-1].paths, parts[0].classes
        S = sum(p.S for p in parts) / len(parts)
        self.threshold, self.calib = matching.calibrate(S, self.names, parts[0].groups, keep_known)

    def sims(self, vecs):
        return sum(p.sims(vecs) for p in self.parts) / len(self.parts)

    def query(self, vecs):
        return matching.rank(self.sims(vecs), self.names, self.paths)

    @property
    def sources(self):
        return {p.source for p in self.parts}


def sources_of(matcher):
    return matcher.sources if isinstance(matcher, Combined) else {matcher.source}


def log(rows: list[dict]):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    now = datetime.now().isoformat(timespec="seconds")
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({"time": now, **row})


def session_stats():
    """Score each matcher against the recorded true answers."""
    if not LOG.exists():
        return {"uploads": 0}
    rows = list(csv.DictReader(LOG.open()))
    preds = {(r["upload_id"], r["method"]): r for r in rows if r["event"] == "identify"}
    truth = {r["upload_id"]: r for r in rows if r["event"] == "feedback"}  # last wins
    stats = {"uploads": len({u for u, _ in preds}), "scored": len(truth), "bad_photo": 0, "methods": {}}
    for uid, fb in truth.items():
        if fb["verdict"] == "bad_photo":
            stats["bad_photo"] += 1
            continue
        for (u, method), p in preds.items():
            if u != uid:
                continue
            m = stats["methods"].setdefault(method, dict.fromkeys(
                ["known", "top1", "top5", "known_flagged_new", "new", "new_flagged"], 0))
            names = [x.rsplit(":", 1)[0] for x in p["top5"].split(";")]
            flagged = p["likely_new"] == "1"
            if fb["verdict"] == "known":
                m["known"] += 1
                m["top1"] += names[0] == fb["true_name"]
                m["top5"] += fb["true_name"] in names
                m["known_flagged_new"] += flagged
            else:
                m["new"] += 1
                m["new_flagged"] += flagged
    return stats


def build_app(matchers: dict, use_gemini: bool = True, sift=None) -> FastAPI:
    app = FastAPI()
    UPLOADS.mkdir(exist_ok=True)
    app.mount("/data", StaticFiles(directory=identify.DATA), name="data")
    app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
    crop_client = shellcrop.make_client()
    pool = ThreadPoolExecutor(4)
    embed_pool = ThreadPoolExecutor(4)  # crop variants embed in parallel once the box is known
    needed = set().union(*(sources_of(m) for m in matchers.values()))
    crop_variants = [v for v in shellcrop.VARIANTS if v in needed]
    classes = sorted(set().union(*(m.classes for m in [*matchers.values(), *([sift] if sift else [])])))
    if sift and "combined" in matchers:
        assert sift.ids == matchers["combined"].parts[0].ids, "SIFT and embedding references differ"
        sift_cidx = np.array([sift.classes.index(n) for n in sift.names])

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/crops", response_class=HTMLResponse)
    def crops_page():
        return crops_review_html()

    @app.get("/api/info")
    def info():
        shown = {**({"sift": sift} if sift else {}), **matchers}
        return {"classes": classes, "stats": session_stats(),
                "methods": {k: {"n_ref": len(m.names), "threshold": round(m.threshold, 3),
                                "calib": m.calib} for k, m in shown.items()}}

    def run_crops(path: Path, img: Image.Image):
        """One box call, then save and embed every crop variant (and SIFT the shell crop) in parallel."""
        box = shellcrop.detect_box(crop_client, img)
        out, sift_job = {}, None
        for v in crop_variants:
            cpath = path.with_name(f"{path.stem}_{v}.jpg")
            crop = shellcrop.crop_to_box(img, box, shellcrop.VARIANTS[v]["margin"]) if box else img
            crop.save(cpath, "JPEG", quality=90)
            out[v] = (cpath, embed_pool.submit(identify.embed_image, cpath))
            if v == "crop" and sift:
                sift_job = embed_pool.submit(sift.query_scores, cpath)
        return box, out, sift_job

    @app.post("/api/identify")
    def identify_upload(image: UploadFile = File(...)):
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(image.file.read())))
        except Exception:
            raise HTTPException(400, "could not read that image")
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        upload_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        path = UPLOADS / f"{upload_id}.jpg"
        img.save(path, "JPEG", quality=85)

        # embeddings per source, run concurrently: whole photo || (box -> crops)
        full_job = pool.submit(identify.embed_image, path) if "full" in needed else None
        crops_job = pool.submit(run_crops, path, img) if crop_variants else None
        vecs, images, errors, box, sift_job = {}, {"full": f"/uploads/{path.name}"}, {}, None, None
        if full_job:
            try:
                vecs["full"] = full_job.result()
            except Exception as e:
                errors["full"] = str(e)[:300]
        if crops_job:
            try:
                box, crops, sift_job = crops_job.result()
                for v, (cpath, job) in crops.items():
                    images[v] = f"/uploads/{cpath.name}"
                    try:
                        vecs[v] = job.result()
                    except Exception as e:
                        errors[v] = str(e)[:300]
            except Exception as e:  # box call failed
                errors.update({v: str(e)[:300] for v in crop_variants})

        results, log_rows = {}, []
        s_sift = None
        if sift:
            try:
                if sift_job is None:
                    raise RuntimeError(errors.get("crop", "no shell crop"))
                s_sift = sift_job.result()
                m = sift.rank(s_sift)
                results["sift"] = {"image": images["crop"], "box": box, "matches": m,
                                   "likely_new": m[0]["sim"] < sift.threshold, "threshold": sift.threshold,
                                   "scale": "spot-match score"}
                log_rows.append({"upload_id": upload_id, "event": "identify", "method": "sift",
                                 "filename": image.filename, "box": box or "",
                                 "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                                 "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                                 "likely_new": int(results["sift"]["likely_new"])})
            except Exception as e:
                results["sift"] = {"error": f"SIFT failed: {str(e)[:250]}"}
        comb = matchers.get("combined")
        if s_sift is not None and comb and not (sources_of(comb) - vecs.keys()):
            results = {"best": best_guess(s_sift, comb.sims(vecs), images["crop"], box), **results}
            b = results["best"]
            log_rows.insert(0, {"upload_id": upload_id, "event": "identify", "method": "best",
                                "filename": image.filename, "box": "",
                                "top1": b["matches"][0]["name"], "top1_sim": b["matches"][0]["sim"],
                                "top5": ";".join(f"{x['name']}:{x['sim']}" for x in b["matches"]),
                                "likely_new": int(b["likely_new"]), "notes": f"spot {b['spot']}"})
        for method, matcher in matchers.items():
            missing = [s for s in sources_of(matcher) if s not in vecs]
            if missing:
                results[method] = {"error": errors.get(missing[0], "embedding failed")}
                continue
            m = matcher.query(vecs)
            r = {"image": images[matcher.display], "box": box if matcher.display != "full" else None,
                 "matches": m, "likely_new": m[0]["sim"] < matcher.threshold,
                 "threshold": round(matcher.threshold, 3)}
            results[method] = r
            log_rows.append({"upload_id": upload_id, "event": "identify", "method": method,
                             "filename": image.filename, "box": r["box"] or "",
                             "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                             "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                             "likely_new": int(r["likely_new"])})
        shortlist = next((results[k] for k in ("best", "combined") if "matches" in results.get(k, {})), None)
        if use_gemini and shortlist and "crop" in images:
            g = gemini_pick(shortlist, images["crop"], upload_id)
            results = {k: v for k, v in [("best", results.get("best")), ("gemini", g),
                                         ("sift", results.get("sift"))] if v} | results
            g = results["gemini"]
            if "matches" in g:
                m = g["matches"]
                log_rows.insert(0, {"upload_id": upload_id, "event": "identify", "method": "gemini",
                                    "filename": image.filename, "box": g["box"] or "",
                                    "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                                    "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                                    "likely_new": int(g["likely_new"]), "notes": g["reason"]})
        log(log_rows)
        return {"upload_id": upload_id, "image": f"/uploads/{path.name}", "methods": results}

    def best_guess(s_sift, s_emb, image_url, box, top=5, per_class=3):
        """Fuse SIFT and combined-embedding evidence per turtle (matching.fuse)."""
        fused = matching.fuse(s_sift, s_emb, sift_cidx, len(sift.classes))
        matches = []
        for k in np.argsort(-fused)[:top]:
            mine = np.flatnonzero(sift_cidx == k)
            mine = mine[np.lexsort((-s_emb[mine], -s_sift[mine]))][:per_class]
            matches.append({"name": sift.classes[k], "sim": round(float(fused[k]), 2),
                            "refs": [{"path": sift.paths[j], "sim": round(float(s_sift[j]), 1)} for j in mine]})
        spot = round(float(s_sift[sift_cidx == sift.classes.index(matches[0]["name"])].max()), 1)
        return {"image": image_url, "box": box, "matches": matches, "spot": spot,
                "likely_new": spot < sift.threshold, "threshold": sift.threshold}

    def gemini_pick(shortlist: dict, query_url: str, upload_id: str):
        """Gemini chooses among a matcher's top 5 by comparing shell crops."""
        by_name = {m["name"]: m for m in shortlist["matches"]}
        candidates = {n: [ROOT / r["path"].replace("data/crops_tight/", "data/crops/", 1) for r in m["refs"]]
                      for n, m in by_name.items()}
        try:
            out = rerank.rerank(crop_client, UPLOADS / Path(query_url).name, candidates, seed=hash(upload_id))
        except Exception as e:
            return {"error": f"Gemini re-rank failed: {str(e)[:250]}"}
        # The "weak match" flag stays with the shortlist's rule: Gemini always picks someone.
        picked = {**shortlist, "matches": [by_name[n] for n in out["ranking"]], "reason": out["reason"]}
        if "spot" in shortlist:  # best-guess refs carry SIFT scores: re-check Gemini's pick
            picked["spot"] = max(r["sim"] for r in picked["matches"][0]["refs"])
            picked["likely_new"] = picked["spot"] < shortlist["threshold"]
        return picked

    @app.post("/api/feedback")
    def feedback(upload_id: str = Form(...), verdict: str = Form(...),
                 true_name: str = Form(""), notes: str = Form("")):
        if verdict not in VERDICTS:
            raise HTTPException(400, "unknown verdict")
        if verdict == "known" and not true_name:
            raise HTTPException(400, "known turtle needs a name")
        log([{"upload_id": upload_id, "event": "feedback", "verdict": verdict,
              "true_name": true_name, "notes": notes}])
        return {"stats": session_stats()}

    return app


def crops_review_html():
    """Grid of every reference crop, grouped by turtle; misses first."""
    boxes = identify.DATA / "crops.csv"
    if not boxes.exists():
        return "<p>No crops yet — run <code>python crop_photos.py</code>.</p>"
    rows = sorted(csv.DictReader(boxes.open()),
                  key=lambda r: (r["found"] == "1", r["turtle_name"], int(r["capture_id"])))
    esc = lambda t: html.escape(str(t), quote=True)
    tiles, current = [], None
    for r in rows:
        group = r["turtle_name"] if r["found"] == "1" else "No shell found (full frame kept)"
        if group != current:
            tiles.append(f"<h2>{esc(group)}</h2>")
            current = group
        rel = f"{esc(r['turtle_name'])}/{esc(r['capture_id'])}.jpg"
        imgs = "".join(f'<img loading="lazy" src="/data/{cfg["dir"]}/{rel}" title="{v}">'
                       for v, cfg in shellcrop.VARIANTS.items())
        tiles.append(f'<figure><a href="/data/{rel}" target="_blank" title="open original">'
                     f'{imgs}</a><figcaption>{esc(r["capture_id"])}'
                     f'{"" if r["found"] == "1" else " · " + esc(r["turtle_name"])}</figcaption></figure>')
    n_miss = sum(r["found"] != "1" for r in rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference Crops</title>
<style>body{{margin:0;padding:16px;font:14px/1.4 system-ui,sans-serif;background:#f6f4ee;color:#1d2a22}}
h1{{font-size:20px;margin:0 0 4px}}h2{{width:100%;font-size:15px;margin:18px 0 6px}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}figure{{margin:0}}
figure a{{display:flex;gap:3px}}img{{height:110px;border-radius:6px;display:block}}figcaption{{color:#6b756e;font-size:12px}}</style></head>
<body><h1>Reference crops</h1><div>{len(rows)} photos · {n_miss} with no shell found ·
each pair is crop | tight · click to open the original</div><div class="grid">{"".join(tiles)}</div></body></html>"""


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turtle ID</title>
<style>
:root{--bg:#f6f4ee;--card:#fff;--ink:#1d2a22;--muted:#6b756e;--line:#e2ded3;--accent:#2f6b4f;--warn:#b5651d;--ok:#2f6b4f;--bad:#a33;--hit:#e5f0ea}
*{box-sizing:border-box}body{margin:0;font:16px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0 2px}.sub{color:var(--muted);font-size:13px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
label.drop{display:block;border:2px dashed var(--line);border-radius:12px;padding:22px;text-align:center;cursor:pointer;color:var(--muted)}
label.drop b{color:var(--accent)}input[type=file]{display:none}
.query{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}.query>img{width:200px;max-width:100%;border-radius:8px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.col h2{font-size:16px;margin:0 0 8px}.cropimg{display:block;margin-bottom:8px}.cropimg img{max-height:180px;max-width:100%;border-radius:8px}
.banner{padding:8px 10px;border-radius:8px;font-weight:600;font-size:14px;margin:6px 0}
.banner.new{background:#fbeee0;color:var(--warn)}.banner.known{background:var(--hit);color:var(--ok)}
.match{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center;border-top:1px solid var(--line);padding:8px 0}
.match.truth{background:var(--hit);margin:0 -8px;padding:8px}
.match .name{font-weight:600}.match .sim{color:var(--muted);font-size:13px}
.refs{grid-column:1/-1;display:flex;gap:6px;overflow-x:auto}.refs img{height:84px;border-radius:6px}
button{font:inherit;font-size:14px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 10px;cursor:pointer}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}select,input[type=text]{font:inherit;padding:7px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.done{color:var(--ok);font-weight:600}.err{color:var(--bad)}.muted{color:var(--muted);font-size:13px}
</style></head><body><main>
<h1>Turtle ID</h1>
<div class="sub"><span id="meta">loading…</span> · <a href="/crops" target="_blank">review reference crops</a></div>
<div class="card">
  <label class="drop"><input id="file" type="file" accept="image/*"><b>Choose or take a photo</b><br>top of the shell, filling the frame</label>
</div>
<div id="result"></div>
<div class="card" id="stats"></div>
</main><script>
let INFO, CUR, TRUTH;
const LABEL = {best: 'SIFT + embeddings (best)', gemini: 'Gemini pick (second opinion)', sift: 'Spot match (SIFT)', combined: 'Combined', full: 'Whole photo', crop: 'Cropped to shell', tight: 'Tight (inside shell)'};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (a, b) => b ? `${a}/${b} (${Math.round(100*a/b)}%)` : '–';
function showStats(s){
  if(!s || !s.uploads){ $('stats').innerHTML = '<span class="muted">No uploads yet this session.</span>'; return; }
  const rows = Object.entries(s.methods || {}).map(([m, x]) => `<tr><td>${LABEL[m]||m}</td>
    <td>${pct(x.top1, x.known)}</td><td>${pct(x.top5, x.known)}</td>
    <td>${pct(x.new_flagged, x.new)}</td><td>${pct(x.known_flagged_new, x.known)}</td></tr>`).join('');
  $('stats').innerHTML = `<div class="muted" style="margin-bottom:6px">Session: ${s.uploads} uploaded, ${s.scored} scored${s.bad_photo ? `, ${s.bad_photo} bad photos` : ''}</div>
    <table><tr><th>matcher</th><th>top-1 right</th><th>in top 5</th><th>new turtles flagged weak</th><th>known flagged weak</th></tr>${rows}</table>`;
}
async function load(){
  INFO = await (await fetch('/api/info')).json();
  $('meta').textContent = `${INFO.classes.length} known turtles · ` + Object.entries(INFO.methods).map(([m, x]) =>
    `${LABEL[m]}: ${x.n_ref} refs, "new" below ${x.threshold}`).join(' · ');
  showStats(INFO.stats);
}
$('file').onchange = async e => {
  const f = e.target.files[0]; if(!f) return;
  $('result').innerHTML = '<div class="card">Cropping, embedding and matching…</div>';
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/api/identify', {method:'POST', body:fd});
  e.target.value = '';
  if(!r.ok){ $('result').innerHTML = `<div class="card err">${esc((await r.json()).detail || 'failed')}</div>`; return; }
  CUR = await r.json(); TRUTH = null; render();
};
function column(method, r){
  if(r.error) return `<div class="card col"><h2>${LABEL[method]}</h2><p class="err">${esc(r.error)}</p></div>`;
  const m = r.matches;
  const spot = r.spot ?? m[0].sim;
  const banner = (method === 'sift' || r.spot !== undefined)
    ? (r.likely_new
      ? `<div class="banner new">No confirming spot match (score ${spot} &lt; ${r.threshold}): could be a new turtle, or a view we don't have</div>`
      : `<div class="banner known">Confirmed by spot match: ${esc(m[0].name)} (score ${spot})</div>`)
    : r.likely_new
    ? `<div class="banner new">Weak match, could be a new turtle: best ${m[0].sim}, cut-off ${r.threshold}</div>`
    : `<div class="banner known">Best: ${esc(m[0].name)} (${m[0].sim})</div>`;
  const thumb = method !== 'full' ? `<a class="cropimg" href="${r.image}" target="_blank"><img src="${r.image}" title="${r.box ? 'box ' + r.box : 'no shell found — full frame'}"></a>` : '';
  const rows = m.map((x, i) => `<div class="match ${TRUTH === x.name ? 'truth' : ''}">
      <div><span class="name">${i+1}. ${esc(x.name)}</span> <span class="sim">${x.sim}</span></div>
      <button data-name="${esc(x.name)}" onclick="send('known', this.dataset.name)">This is it</button>
      <div class="refs">${x.refs.map(rf => `<img loading="lazy" src="/${esc(rf.path)}" title="${rf.sim}">`).join('')}</div>
    </div>`).join('');
  const reason = r.reason ? `<p class="muted">Gemini: ${esc(r.reason)}</p>` : '';
  return `<div class="card col"><h2>${LABEL[method]}${method !== 'full' && !r.box ? ' <span class="muted">(no shell found)</span>' : ''}</h2>${thumb}${banner}${reason}${rows}</div>`;
}
function render(){
  const opts = INFO.classes.map(c => `<option>${esc(c)}</option>`).join('');
  $('result').innerHTML = `<div class="card"><div class="query"><img src="${CUR.image}">
      <div style="flex:1;min-width:240px"><b>What is it really?</b>
        <div class="muted">Tap "This is it" on the right turtle below, or:</div>
        <div class="row"><select id="truename"><option value="">Not in either list — pick…</option>${opts}</select>
          <button onclick="send('known', $('truename').value)">Save</button></div>
        <div class="row"><button onclick="send('new_turtle')">New turtle</button><button onclick="send('bad_photo')">Bad photo</button></div>
        <div class="row"><input type="text" id="notes" placeholder="notes (optional)" style="flex:1"></div>
        <div id="saved"></div></div></div></div>
    <div class="cols">${Object.entries(CUR.methods).map(([m, r]) => column(m, r)).join('')}</div>`;
}
async function send(verdict, name){
  if(verdict === 'known' && !name){ $('truename').focus(); return; }
  const notes = $('notes').value;
  const fd = new FormData();
  fd.append('upload_id', CUR.upload_id); fd.append('verdict', verdict);
  fd.append('true_name', name || ''); fd.append('notes', notes);
  const r = await (await fetch('/api/feedback', {method:'POST', body:fd})).json();
  TRUTH = verdict === 'known' ? name : null;
  render(); $('notes').value = notes;
  $('saved').innerHTML = `<p class="done">Saved: ${verdict === 'known' ? esc(name) : verdict.replace('_', ' ')}</p>`;
  showStats(r.stats);
}
load();
</script></body></html>
"""


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pca", type=int, default=identify.DEFAULT_PCA)
    ap.add_argument("--db", type=Path, default=identify.DB)
    for v, cfg in shellcrop.VARIANTS.items():
        ap.add_argument(f"--{v}-db", type=Path, default=identify.DATA / cfg["db"])
    ap.add_argument("--no-crop", action="store_true", help="whole-photo matcher only")
    ap.add_argument("--no-gemini", action="store_true", help="skip the Gemini re-rank column")
    ap.add_argument("--no-sift", action="store_true", help="skip the SIFT spot-match column")
    ap.add_argument("--keep-known", type=float, default=0.8,
                    help="weak-match cut-off still recognises this share of known turtles (default 0.8)")
    args = ap.parse_args()

    if not args.db.exists():
        ap.error(f"missing {args.db} — run embed_photos.py first")
    identify.load_env()
    if not matching.META.exists():
        print(f"no {matching.META} — run fetch_meta.py for other-day calibration; using per-photo groups")

    kk = args.keep_known
    pca = {"full": Index("full", args.db, args.pca, keep_known=kk)}
    lda = {}
    if not args.no_crop:
        for v in shellcrop.VARIANTS:
            db = getattr(args, f"{v}_db")
            if db.exists():
                pca[v] = Index(v, db, args.pca, keep_known=kk)
                lda[v] = Index(v, db, args.pca, lda=True, keep_known=kk)
            else:
                print(f"no {db} — run crop_photos.py for the {v!r} matcher")
    matchers = {}
    if {"crop", "tight"} <= lda.keys():
        matchers["combined"] = Combined([lda["crop"], lda["tight"]], keep_known=kk)
    matchers.update(pca)
    for k, m in matchers.items():
        print(f"{k}: {len(m.names)} reference photos, {len(m.classes)} turtles; "
              f"weak-match cut-off {m.threshold:.3f} {m.calib}")
    ip = lan_ip()
    print(f"open http://localhost:{args.port}" + (f"  (phone on same Wi-Fi: http://{ip}:{args.port})" if ip else ""))
    sift = None
    if not args.no_sift and "crop" in pca:
        ref = pca["crop"]
        sift = sift_match.SiftIndex(ref.names, ref.ids, ref.paths, identify.DATA / "sift_crop250.pkl", root=ROOT)
        print(f"sift: {len(sift.names)} reference crops; score >= {sift.threshold:g} confirms a known turtle")
    use_gemini = not args.no_gemini and "combined" in matchers
    if use_gemini:
        print(f"gemini: re-ranks the combined top 5 with {rerank.RERANK_MODEL}")
    uvicorn.run(build_app(matchers, use_gemini, sift), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
