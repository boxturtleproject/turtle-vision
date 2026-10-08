"""Local web app for field-testing turtle identification.

Upload a carapace photo (from a laptop or a phone on the same Wi-Fi), see the
closest known individuals with their reference photos, and record whether the
match was right. Every upload and verdict is appended to
results/session_log.csv so a test session can be scored afterwards.

Matching is the headline classifier from identify.py (PCA -> 128, 1-NN
cosine over all labeled carapace photos). A "likely new turtle" cut-off is
calibrated at startup by leave-one-out over the reference set: for each
reference photo, how similar is its nearest same-turtle photo vs. its nearest
other-turtle photo.

    python app.py                 # http://localhost:8000 (+ LAN URL printed)
    python app.py --threshold 0.6 # override the calibrated cut-off
"""
import argparse, csv, io, socket, uuid
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

register_heif_opener()  # iPhone HEIC uploads

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
LOG = ROOT / "results" / "session_log.csv"
LOG_FIELDS = ["time", "upload_id", "event", "filename", "top1", "top1_sim",
              "top5", "likely_new", "verdict", "true_name", "notes"]
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
    def __init__(self, db: Path, pca_dim: int):
        self.names, self.paths, X = identify.load_reference(identify.DROP, db)
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


def log(row: dict):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), **row})


def session_stats():
    if not LOG.exists():
        return {}
    rows = list(csv.DictReader(LOG.open()))
    top1 = {r["upload_id"]: r["top1"] for r in rows if r["event"] == "identify"}
    verdicts = {r["upload_id"]: r for r in rows if r["event"] == "feedback"}  # last wins
    counts = {"uploads": len(top1), "scored": len(verdicts)}
    for r in verdicts.values():
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return counts


def build_app(index: Index) -> FastAPI:
    app = FastAPI()
    UPLOADS.mkdir(exist_ok=True)
    app.mount("/data", StaticFiles(directory=identify.DATA), name="data")
    app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/api/info")
    def info():
        return {"classes": index.classes, "n_ref": len(index.names),
                "threshold": round(index.threshold, 3), "calib": index.calib,
                "stats": session_stats()}

    @app.post("/api/identify")
    async def identify_upload(image: UploadFile = File(...)):
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(await image.read())))
        except Exception:
            raise HTTPException(400, "could not read that image")
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        upload_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        path = UPLOADS / f"{upload_id}.jpg"
        img.save(path, "JPEG", quality=85)

        matches = index.query(identify.embed_image(path))
        likely_new = matches[0]["sim"] < index.threshold
        log({"upload_id": upload_id, "event": "identify", "filename": image.filename,
             "top1": matches[0]["name"], "top1_sim": matches[0]["sim"],
             "top5": ";".join(f"{m['name']}:{m['sim']}" for m in matches),
             "likely_new": int(likely_new)})
        return {"upload_id": upload_id, "image": f"/uploads/{path.name}",
                "matches": matches, "likely_new": likely_new,
                "threshold": round(index.threshold, 3)}

    @app.post("/api/feedback")
    def feedback(upload_id: str = Form(...), verdict: str = Form(...),
                 true_name: str = Form(""), notes: str = Form("")):
        if verdict not in ("correct", "in_top5", "wrong", "new_turtle", "bad_photo"):
            raise HTTPException(400, "unknown verdict")
        log({"upload_id": upload_id, "event": "feedback", "verdict": verdict,
             "true_name": true_name, "notes": notes})
        return {"stats": session_stats()}

    return app


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turtle ID</title>
<style>
:root{--bg:#f6f4ee;--card:#fff;--ink:#1d2a22;--muted:#6b756e;--line:#e2ded3;--accent:#2f6b4f;--warn:#b5651d;--ok:#2f6b4f;--bad:#a33}
*{box-sizing:border-box}body{margin:0;font:16px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0 2px}.sub{color:var(--muted);font-size:13px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
label.drop{display:block;border:2px dashed var(--line);border-radius:12px;padding:26px;text-align:center;cursor:pointer;color:var(--muted)}
label.drop b{color:var(--accent)}input[type=file]{display:none}
.query{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}.query img{width:220px;max-width:100%;border-radius:8px}
.banner{padding:10px 12px;border-radius:8px;font-weight:600;margin:10px 0}
.banner.new{background:#fbeee0;color:var(--warn)}.banner.known{background:#e5f0ea;color:var(--ok)}
.match{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;border-top:1px solid var(--line);padding:10px 0}
.match .name{font-weight:600}.match .sim{color:var(--muted);font-size:13px}
.refs{grid-column:1/-1;display:flex;gap:6px;overflow-x:auto}.refs img{height:96px;border-radius:6px}
button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}select,input[type=text]{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px}
.stats{font-size:14px;color:var(--muted)}.done{color:var(--ok);font-weight:600}.err{color:var(--bad)}
</style></head><body><main>
<h1>Turtle ID</h1>
<div class="sub" id="meta">loading…</div>
<div class="card">
  <label class="drop"><input id="file" type="file" accept="image/*"><b>Choose or take a photo</b><br>top of the shell, filling the frame</label>
</div>
<div id="result"></div>
<div class="card stats" id="stats"></div>
</main><script>
let INFO, CUR;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function showStats(s){
  if(!s || !s.uploads){ $('stats').textContent = 'No uploads yet this session.'; return; }
  const parts = ['correct','in_top5','wrong','new_turtle','bad_photo'].filter(k => s[k]).map(k => `${k.replace('_',' ')}: ${s[k]}`);
  $('stats').textContent = `Session — ${s.uploads} uploaded, ${s.scored} scored` + (parts.length ? ' · ' + parts.join(' · ') : '');
}
async function load(){
  INFO = await (await fetch('/api/info')).json();
  $('meta').textContent = `${INFO.classes.length} known turtles · ${INFO.n_ref} reference photos · "new turtle" below similarity ${INFO.threshold} (keeps ${Math.round(INFO.calib.known_kept*100)}% of known, flags ${Math.round(INFO.calib.new_caught*100)}% of new in calibration)`;
  showStats(INFO.stats);
}
$('file').onchange = async e => {
  const f = e.target.files[0]; if(!f) return;
  $('result').innerHTML = '<div class="card">Embedding and matching…</div>';
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/api/identify', {method:'POST', body:fd});
  e.target.value = '';
  if(!r.ok){ $('result').innerHTML = `<div class="card err">${esc((await r.json()).detail || 'failed')}</div>`; return; }
  CUR = await r.json(); render();
};
function render(){
  const m = CUR.matches;
  const banner = CUR.likely_new
    ? `<div class="banner new">Possibly a new turtle — best match ${m[0].sim} is below ${CUR.threshold}</div>`
    : `<div class="banner known">Best match: ${esc(m[0].name)} (${m[0].sim})</div>`;
  const rows = m.map((x,i) => `<div class="match">
      <div><span class="name">${i+1}. ${esc(x.name)}</span> <span class="sim">similarity ${x.sim}</span></div>
      <button onclick="send('${i===0?'correct':'in_top5'}', ${i})">This is it</button>
      <div class="refs">${x.refs.map(r => `<img loading="lazy" src="/${esc(r.path)}" title="${r.sim}">`).join('')}</div>
    </div>`).join('');
  const opts = INFO.classes.map(c => `<option>${esc(c)}</option>`).join('');
  $('result').innerHTML = `<div class="card">
    <div class="query"><img src="${CUR.image}"><div style="flex:1;min-width:200px">${banner}
      <div class="row"><select id="truename"><option value="">Not listed — pick turtle…</option>${opts}</select>
        <button onclick="send('wrong')">Save as wrong</button></div>
      <div class="row"><button onclick="send('new_turtle')">New turtle</button><button onclick="send('bad_photo')">Bad photo</button></div>
      <div class="row"><input type="text" id="notes" placeholder="notes (optional)" style="flex:1"></div>
      <div id="saved"></div></div></div>
    ${rows}</div>`;
}
async function send(verdict, i){
  let name = '';
  if(i !== undefined) name = CUR.matches[i].name;
  if(verdict === 'wrong'){ name = $('truename').value; if(!name){ $('truename').focus(); return; } }
  const fd = new FormData();
  fd.append('upload_id', CUR.upload_id); fd.append('verdict', verdict);
  fd.append('true_name', name); fd.append('notes', $('notes').value);
  const r = await (await fetch('/api/feedback', {method:'POST', body:fd})).json();
  $('saved').innerHTML = `<p class="done">Saved: ${verdict.replace('_',' ')}${name ? ' — ' + esc(name) : ''}</p>`;
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
    ap.add_argument("--threshold", type=float, help="override the calibrated 'new turtle' cut-off")
    ap.add_argument("--db", type=Path, default=identify.DB)
    args = ap.parse_args()

    if not args.db.exists():
        ap.error(f"missing {args.db} — run embed_photos.py first")
    identify.load_env()

    index = Index(args.db, args.pca)
    if args.threshold is not None:
        index.threshold = args.threshold
    print(f"reference: {len(index.names)} photos, {len(index.classes)} turtles; "
          f"new-turtle cut-off {index.threshold:.3f} {index.calib}")
    ip = lan_ip()
    print(f"open http://localhost:{args.port}" + (f"  (phone on same Wi-Fi: http://{ip}:{args.port})" if ip else ""))
    uvicorn.run(build_app(index), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
