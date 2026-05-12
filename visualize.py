"""Interactive 2D projections of Gemini carapace embeddings.

Computes t-SNE and UMAP projections of every carapace photo (after
upstream filters: photos only, Unidentified excluded) and writes a single
self-contained HTML file with a full-screen scatter plot, t-SNE/UMAP
toggle (default t-SNE), and image-on-hover.

Output: embedding_viz.html (at the repo root, so the relative image paths
        in classifications.csv resolve correctly).
"""
import colorsys, csv, html, json, sqlite3, struct, sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
import umap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "embeddings.sqlite"
SRC = DATA / "classifications.csv"
OUT = ROOT / "embedding_viz.html"
DIM = 3072
SEED = 42


def load_carapace():
    rows = list(csv.DictReader(SRC.open()))
    rows = [r for r in rows
            if r["category"] == "carapace"
            and r["media_type"] == "photo"
            and r["turtle_name"] != "Unidentified"]
    ids = [int(r["capture_id"]) for r in rows]
    con = sqlite3.connect(DB)
    rows_db = con.execute(
        f"SELECT capture_id, embedding FROM images "
        f"WHERE capture_id IN ({','.join('?'*len(ids))})", ids
    ).fetchall()
    emb = {cid: np.array(struct.unpack(f"<{DIM}f", b), dtype=np.float32)
           for cid, b in rows_db}
    keep = [r for r in rows if int(r["capture_id"]) in emb]
    missing = len(rows) - len(keep)
    if missing:
        print(f"warning: {missing} carapace photos missing embeddings", file=sys.stderr)
    X = np.stack([emb[int(r["capture_id"])] for r in keep]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    names = [r["turtle_name"] for r in keep]
    paths = [r["file_path"] for r in keep]
    return keep, names, paths, X


def palette(n):
    """n visually-distinct hex colors via even-spaced HSV."""
    out = []
    for i in range(n):
        # alternate saturation/value to break up neighbors a bit
        h = (i * 360 / n) / 360.0
        s = 0.55 + 0.20 * (i % 2)
        v = 0.95 - 0.15 * ((i // 2) % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append("#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)))
    return out


def project(X):
    print(f"running t-SNE on {X.shape[0]} × {X.shape[1]}…", file=sys.stderr)
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                metric="cosine", random_state=SEED).fit_transform(X)
    print(f"running UMAP on {X.shape[0]} × {X.shape[1]}…", file=sys.stderr)
    um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=SEED).fit_transform(X)
    return tsne, um


HTML_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Box-turtle carapace embedding projections</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; }
  body { font-family: system-ui, sans-serif; color: #222; background: #fff;
         display: flex; flex-direction: column; overflow: hidden; }
  header { flex: 0 0 auto; padding: 8px 14px; border-bottom: 1px solid #ddd;
           display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  h1 { font-size: 14px; margin: 0; font-weight: 600; }
  .sub { font-size: 11px; color: #666; }
  .switcher { display: inline-flex; border: 1px solid #888; border-radius: 4px;
              overflow: hidden; }
  .switcher button { padding: 4px 12px; cursor: pointer; border: 0;
                     background: #fff; font-size: 12px; }
  .switcher button + button { border-left: 1px solid #888; }
  .switcher button.active { background: #333; color: #fff; }
  .hint { font-size: 11px; color: #666; }
  main { flex: 1 1 auto; display: flex; min-height: 0; }
  .plot-area { flex: 1 1 auto; min-width: 0; min-height: 0; position: relative; }
  svg.plot { width: 100%; height: 100%; display: block; background: #fff; }
  circle { opacity: 0.78; }
  circle:hover { stroke: #000; stroke-width: 1.5; opacity: 1; }
  circle.dim { opacity: 0.06; }
  circle.hl  { opacity: 1.0; stroke: #000; stroke-width: 1.0; }
  #tooltip {
    position: fixed; pointer-events: none; z-index: 10;
    background: #fff; border: 1px solid #888; padding: 6px;
    font-size: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    display: none; max-width: 360px;
  }
  #tooltip img { display: block; max-width: 340px; max-height: 340px; }
  #tooltip .name { font-weight: 600; margin-bottom: 4px; }
  .legend { width: 200px; flex: 0 0 200px; overflow-y: auto; padding: 8px 10px;
            border-left: 1px solid #ddd; font-size: 11px; line-height: 1.5; }
  .legend .item { white-space: nowrap; cursor: pointer;
                  padding: 1px 4px; border-radius: 2px; user-select: none; }
  .legend .item:hover { background: #eee; }
  .legend .item.active { background: #ffe8a8; }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                margin-right: 6px; vertical-align: middle; border: 1px solid #0002; }
</style>
</head>
<body>
<header>
  <h1>Box-turtle carapace embeddings</h1>
  <span class="sub">__SUB__</span>
  <div class="switcher" role="tablist">
    <button id="btn-t" class="active" data-key="t">t-SNE</button>
    <button id="btn-u" data-key="u">UMAP</button>
  </div>
  <span class="hint">Hover a point for image preview · click a name to isolate.</span>
</header>
<main>
  <div class="plot-area">
    <svg id="plot" class="plot"></svg>
  </div>
  <div class="legend" id="legend"></div>
</main>
<div id="tooltip"></div>
<script>
const DATA = __DATA_JSON__;
const NS = "http://www.w3.org/2000/svg";
const PAD = 24;
let currentKey = "t";   // 't' or 'u'

const svg = document.getElementById("plot");
const circles = [];
DATA.points.forEach((p, i) => {
  const c = document.createElementNS(NS, "circle");
  c.setAttribute("r", 4.5);
  c.setAttribute("fill", DATA.colors[p.c]);
  c.dataset.idx = i;
  c.dataset.cls = p.c;
  svg.appendChild(c);
  circles.push(c);
});

function layout() {
  const rect = svg.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  if (W < 2 || H < 2) return;
  const xs = DATA.points.map(p => p[currentKey][0]);
  const ys = DATA.points.map(p => p[currentKey][1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (W - 2*PAD) / (xmax - xmin || 1);
  const sy = (H - 2*PAD) / (ymax - ymin || 1);
  for (let i = 0; i < DATA.points.length; i++) {
    const p = DATA.points[i];
    const cx = PAD + (p[currentKey][0] - xmin) * sx;
    const cy = H - PAD - (p[currentKey][1] - ymin) * sy;  // flip y
    circles[i].setAttribute("cx", cx);
    circles[i].setAttribute("cy", cy);
  }
}

window.addEventListener("resize", layout);

// Projection switcher
document.querySelectorAll(".switcher button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".switcher button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentKey = btn.dataset.key;
    layout();
  });
});

// Hover tooltip
const tt = document.getElementById("tooltip");
function showTT(i, ev) {
  const p = DATA.points[i];
  tt.innerHTML =
    '<div class="name">' + DATA.classes[p.c] + ' · capture ' + p.id + '</div>' +
    '<img src="' + p.path + '" loading="eager" />';
  tt.style.display = "block";
  positionTT(ev);
}
function positionTT(ev) {
  const pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const r = tt.getBoundingClientRect();
  if (x + r.width > window.innerWidth) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight) y = ev.clientY - r.height - pad;
  tt.style.left = x + "px"; tt.style.top = y + "px";
}
function hideTT() { tt.style.display = "none"; }
document.addEventListener("mouseover", e => {
  if (e.target.tagName === "circle") showTT(+e.target.dataset.idx, e);
});
document.addEventListener("mousemove", e => {
  if (tt.style.display === "block") positionTT(e);
});
document.addEventListener("mouseout", e => {
  if (e.target.tagName === "circle") hideTT();
});

// Legend with click-to-isolate
let activeClass = -1;
function applyHighlight() {
  for (const c of circles) {
    c.classList.remove("dim"); c.classList.remove("hl");
    if (activeClass >= 0) {
      if (+c.dataset.cls === activeClass) c.classList.add("hl");
      else c.classList.add("dim");
    }
  }
  document.querySelectorAll(".legend .item").forEach(el => {
    el.classList.toggle("active", +el.dataset.cls === activeClass);
  });
}
const legend = document.getElementById("legend");
DATA.classes.forEach((name, ci) => {
  const el = document.createElement("div");
  el.className = "item";
  el.dataset.cls = ci;
  el.innerHTML =
    '<span class="sw" style="background:' + DATA.colors[ci] + '"></span>' +
    name + ' <span style="color:#888">(' + DATA.counts[ci] + ')</span>';
  el.addEventListener("click", () => {
    activeClass = (activeClass === ci) ? -1 : ci;
    applyHighlight();
  });
  legend.appendChild(el);
});

// Initial paint after layout has actual size
requestAnimationFrame(layout);
</script>
</body>
</html>
"""


def main():
    rows, names, paths, X = load_carapace()
    print(f"{len(rows)} carapace photos, {len(set(names))} individuals", file=sys.stderr)

    tsne, umap_xy = project(X)

    classes = sorted(set(names))
    cls_idx = {c: i for i, c in enumerate(classes)}
    counts = Counter(names)
    pal = palette(len(classes))

    points = []
    for r, t, u in zip(rows, tsne, umap_xy):
        points.append({
            "id": int(r["capture_id"]),
            "c":  cls_idx[r["turtle_name"]],
            "path": r["file_path"],
            "t": [round(float(t[0]), 4), round(float(t[1]), 4)],
            "u": [round(float(u[0]), 4), round(float(u[1]), 4)],
        })

    payload = {
        "classes": classes,
        "colors":  pal,
        "counts":  [counts[c] for c in classes],
        "points":  points,
    }

    sub = (f"{len(points)} carapace photos · {len(classes)} individuals · "
           f"gemini-embedding-2-preview ({DIM} dims)")
    out_html = (
        HTML_TMPL
        .replace("__SUB__", html.escape(sub))
        .replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    )
    OUT.write_text(out_html)
    size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(ROOT)} ({size/1024:.1f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
