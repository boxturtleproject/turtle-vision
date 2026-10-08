"""Local web app for field-testing turtle identification.

Upload a carapace photo (from a laptop or a phone on the same Wi-Fi) and see
the closest known individuals from two matchers side by side:

  full  — embed the whole photo (the headline classifier from identify.py)
  crop  — crop to the shell with a Gemini bounding box first, then embed;
          compared against crop_photos.py's cropped reference set

Both use PCA -> 128 + 1-NN cosine. You record the true answer once (which
turtle / new turtle / bad photo) and the app scores both matchers against it.
Every upload and verdict is appended to results/session_log.csv.

A "likely new turtle" cut-off is calibrated per matcher at startup by
leave-one-out over its reference set: for each reference photo, how similar
is its nearest same-turtle photo vs. its nearest other-turtle photo.

    python app.py                 # http://localhost:8000 (+ LAN URL printed)
    python app.py --no-crop       # full-frame matcher only
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
from sklearn.decomposition import PCA

import identify
import shellcrop

register_heif_opener()  # iPhone HEIC uploads

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
LOG = ROOT / "results" / "session_log.csv"
CROP_DB = identify.DATA / "embeddings_crop.sqlite"
LOG_FIELDS = ["time", "upload_id", "event", "method", "filename", "box", "top1",
              "top1_sim", "top5", "likely_new", "verdict", "true_name", "notes"]
VERDICTS = ("known", "new_turtle", "bad_photo")
MAX_SIDE = 1280  # match the reference photos (box-turtle-id display derivatives)


def calibrate(X, y):
    """Pick the similarity cut-off that best separates known from new turtles.

    s_known: each photo's best match among *other photos of the same turtle*
    s_new:   each photo's best match among *other turtles* (as if it were new)
    Maximize the mean of P(s_known >= t) and P(s_new < t).
    """
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    same = y[:, None] == y[None, :]
    s_known = np.where(same, S, -np.inf).max(1)
    s_new = np.where(~same, S, -np.inf).max(1)
    s_known = s_known[np.isfinite(s_known)]
    cands = np.unique(np.concatenate([s_known, s_new]))
    kept = (s_known[None, :] >= cands[:, None]).mean(1)
    caught = (s_new[None, :] < cands[:, None]).mean(1)
    i = int(np.argmax(kept + caught))
    return float(cands[i]), {"known_kept": float(kept[i]), "new_caught": float(caught[i])}


class Index:
    def __init__(self, db: Path, pca_dim: int, crops: bool = False):
        self.names, self.paths, X = identify.load_reference(identify.DROP, db)
        if crops:  # show the cropped reference photos in the crop column
            self.paths = [p.replace("data/", "data/crops/", 1) for p in self.paths]
        self.classes = sorted(set(self.names))
        self.pca = PCA(n_components=min(pca_dim, *X.shape), svd_solver="full",
                       random_state=0).fit(X)
        self.X = self._project(X)
        self.threshold, self.calib = calibrate(self.X, np.array(self.names))

    def _project(self, X):
        P = self.pca.transform(X)
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)

    def query(self, v, top=5, per_class=3):
        sim = self.X @ self._project(v[None, :])[0]
        hits: dict[str, list] = {}
        for i in np.argsort(-sim):
            n = self.names[i]
            if n not in hits and len(hits) >= top:
                continue
            refs = hits.setdefault(n, [])
            if len(refs) < per_class:
                refs.append({"path": self.paths[i], "sim": round(float(sim[i]), 3)})
        return [{"name": n, "sim": refs[0]["sim"], "refs": refs} for n, refs in hits.items()]


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


def build_app(indexes: dict[str, Index]) -> FastAPI:
    app = FastAPI()
    UPLOADS.mkdir(exist_ok=True)
    app.mount("/data", StaticFiles(directory=identify.DATA), name="data")
    app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
    crop_client = shellcrop.make_client()
    pool = ThreadPoolExecutor(4)
    classes = sorted(set().union(*(ix.classes for ix in indexes.values())))

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/crops", response_class=HTMLResponse)
    def crops_page():
        return crops_review_html()

    @app.get("/api/info")
    def info():
        return {"classes": classes, "stats": session_stats(),
                "methods": {m: {"n_ref": len(ix.names), "threshold": round(ix.threshold, 3),
                                "calib": ix.calib} for m, ix in indexes.items()}}

    def run_full(path: Path):
        return {"image": f"/uploads/{path.name}", "box": None,
                "matches": indexes["full"].query(identify.embed_image(path))}

    def run_crop(path: Path, img: Image.Image):
        box = shellcrop.detect_box(crop_client, img)
        cpath = path.with_name(path.stem + "_crop.jpg")
        (shellcrop.crop_to_box(img, box) if box else img).save(cpath, "JPEG", quality=90)
        return {"image": f"/uploads/{cpath.name}", "box": box,
                "matches": indexes["crop"].query(identify.embed_image(cpath))}

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

        jobs = {"full": pool.submit(run_full, path)}
        if "crop" in indexes:
            jobs["crop"] = pool.submit(run_crop, path, img)
        results, log_rows = {}, []
        for method, job in jobs.items():
            try:
                r = job.result()
            except Exception as e:
                results[method] = {"error": str(e)[:300]}
                continue
            m = r["matches"]
            r["likely_new"] = m[0]["sim"] < indexes[method].threshold
            r["threshold"] = round(indexes[method].threshold, 3)
            results[method] = r
            log_rows.append({"upload_id": upload_id, "event": "identify", "method": method,
                             "filename": image.filename, "box": r["box"] or "",
                             "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                             "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                             "likely_new": int(r["likely_new"])})
        log(log_rows)
        return {"upload_id": upload_id, "image": f"/uploads/{path.name}", "methods": results}

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
        crop = f"/data/crops/{esc(r['turtle_name'])}/{esc(r['capture_id'])}.jpg"
        orig = f"/data/{esc(r['turtle_name'])}/{esc(r['capture_id'])}.jpg"
        tiles.append(f'<figure><a href="{orig}" target="_blank" title="open original">'
                     f'<img loading="lazy" src="{crop}"></a><figcaption>{esc(r["capture_id"])}'
                     f'{"" if r["found"] == "1" else " · " + esc(r["turtle_name"])}</figcaption></figure>')
    n_miss = sum(r["found"] != "1" for r in rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference Crops</title>
<style>body{{margin:0;padding:16px;font:14px/1.4 system-ui,sans-serif;background:#f6f4ee;color:#1d2a22}}
h1{{font-size:20px;margin:0 0 4px}}h2{{width:100%;font-size:15px;margin:18px 0 6px}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}figure{{margin:0}}
img{{height:120px;border-radius:6px;display:block}}figcaption{{color:#6b756e;font-size:12px}}</style></head>
<body><h1>Reference crops</h1><div>{len(rows)} photos · {n_miss} with no shell found ·
click a crop to open the original</div><div class="grid">{"".join(tiles)}</div></body></html>"""


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
const LABEL = {full: 'Whole photo', crop: 'Cropped to shell'};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (a, b) => b ? `${a}/${b} (${Math.round(100*a/b)}%)` : '–';
function showStats(s){
  if(!s || !s.uploads){ $('stats').innerHTML = '<span class="muted">No uploads yet this session.</span>'; return; }
  const rows = Object.entries(s.methods || {}).map(([m, x]) => `<tr><td>${LABEL[m]||m}</td>
    <td>${pct(x.top1, x.known)}</td><td>${pct(x.top5, x.known)}</td>
    <td>${pct(x.new_flagged, x.new)}</td><td>${pct(x.known_flagged_new, x.known)}</td></tr>`).join('');
  $('stats').innerHTML = `<div class="muted" style="margin-bottom:6px">Session: ${s.uploads} uploaded, ${s.scored} scored${s.bad_photo ? `, ${s.bad_photo} bad photos` : ''}</div>
    <table><tr><th>matcher</th><th>top-1 right</th><th>in top 5</th><th>new turtles flagged</th><th>known wrongly flagged new</th></tr>${rows}</table>`;
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
  const banner = r.likely_new
    ? `<div class="banner new">Possibly new: best ${m[0].sim} &lt; ${r.threshold}</div>`
    : `<div class="banner known">Best: ${esc(m[0].name)} (${m[0].sim})</div>`;
  const thumb = method === 'crop' ? `<a class="cropimg" href="${r.image}" target="_blank"><img src="${r.image}" title="${r.box ? 'box ' + r.box : 'no shell found — full frame'}"></a>` : '';
  const rows = m.map((x, i) => `<div class="match ${TRUTH === x.name ? 'truth' : ''}">
      <div><span class="name">${i+1}. ${esc(x.name)}</span> <span class="sim">${x.sim}</span></div>
      <button data-name="${esc(x.name)}" onclick="send('known', this.dataset.name)">This is it</button>
      <div class="refs">${x.refs.map(rf => `<img loading="lazy" src="/${esc(rf.path)}" title="${rf.sim}">`).join('')}</div>
    </div>`).join('');
  return `<div class="card col"><h2>${LABEL[method]}${method === 'crop' && !r.box ? ' <span class="muted">(no shell found)</span>' : ''}</h2>${thumb}${banner}${rows}</div>`;
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
    ap.add_argument("--crop-db", type=Path, default=CROP_DB)
    ap.add_argument("--no-crop", action="store_true", help="full-frame matcher only")
    args = ap.parse_args()

    if not args.db.exists():
        ap.error(f"missing {args.db} — run embed_photos.py first")
    identify.load_env()

    indexes = {"full": Index(args.db, args.pca)}
    if not args.no_crop:
        if args.crop_db.exists():
            indexes["crop"] = Index(args.crop_db, args.pca, crops=True)
        else:
            print(f"no {args.crop_db} — run crop_photos.py for the cropped matcher; full-frame only")
    for m, ix in indexes.items():
        print(f"{m}: {len(ix.names)} reference photos, {len(ix.classes)} turtles; "
              f"new-turtle cut-off {ix.threshold:.3f} {ix.calib}")
    ip = lan_ip()
    print(f"open http://localhost:{args.port}" + (f"  (phone on same Wi-Fi: http://{ip}:{args.port})" if ip else ""))
    uvicorn.run(build_app(indexes), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
