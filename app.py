"""Local web app for field-testing turtle identification.

Upload a carapace photo, or several photos of one turtle (top/left/right),
from a laptop or a phone on the same Wi-Fi. Several photos are combined into
one answer (each turtle's fused score added up across photos; 0.93 top-1 vs
0.85 for one photo on the different-day test), shown before each photo's own
best guess. A single photo shows these matchers side by side:

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
from fastapi.responses import FileResponse, HTMLResponse
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
TIDY = ROOT / "results" / "uploads_summary.csv"  # one row per uploaded file, rewritten after every change
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


TIDY_FIELDS = ["time", "filename", "upload_id", "set_id", "photos_in_set", "true_answer",
               "set_pick", "set_result", "set_spot_score",
               "best_pick", "best_result", "best_spot_score", "best_confirmed", "best_top5",
               "gemini_pick", "gemini_result", "gemini_reason",
               "sift_pick", "sift_result", "sift_score",
               "embeddings_pick", "embeddings_result", "whole_photo_pick", "whole_photo_result", "notes"]


def write_tidy():
    """results/uploads_summary.csv: one row per uploaded file with every finding."""
    if not LOG.exists():
        return
    rows = list(csv.DictReader(LOG.open()))
    preds = {(r["upload_id"], r["method"]): r for r in rows if r["event"] == "identify"}
    truth = {r["upload_id"]: r for r in rows if r["event"] == "feedback"}  # last wins
    sets = {r["upload_id"]: r["notes"][len("photos "):].split(";") for r in rows
            if r["event"] == "identify" and r["method"] == "sighting"}
    set_of = {pid: sid for sid, pids in sets.items() for pid in pids}

    def answer(uid):
        fb = truth.get(uid)
        if not fb:
            return "", None
        if fb["verdict"] == "known":
            return fb["true_name"], fb["true_name"]
        return fb["verdict"].replace("_", " "), None

    def pick(uid, method, true_name):
        p = preds.get((uid, method))
        if not p:
            return "", "", []
        top5 = [x.rsplit(":", 1)[0] for x in p["top5"].split(";")]
        result = "" if not true_name else ("right" if top5[0] == true_name else
                                            "in top 5" if true_name in top5 else "wrong")
        return top5[0], result, top5

    out = []
    for uid in sorted({r["upload_id"] for r in rows if r["event"] == "identify" and r["method"] != "sighting"}):
        sid = set_of.get(uid, "")
        ans, true_name = answer(sid or uid)
        any_row = next(r for (u, _), r in preds.items() if u == uid)
        best = preds.get((uid, "best"), {})
        spot = best.get("notes", "").replace("spot ", "")
        row = {"time": any_row["time"], "filename": any_row["filename"], "upload_id": uid, "set_id": sid,
               "photos_in_set": len(sets.get(sid, [uid])), "true_answer": ans,
               "best_spot_score": spot,
               "best_confirmed": "" if not best else ("no" if best.get("likely_new") == "1" else "yes"),
               "gemini_reason": preds.get((uid, "gemini"), {}).get("notes", ""),
               "sift_score": preds.get((uid, "sift"), {}).get("top1_sim", ""),
               "notes": (truth.get(sid or uid) or {}).get("notes", "")}
        if sid:
            row["set_pick"], row["set_result"], _ = pick(sid, "sighting", true_name)
            row["set_spot_score"] = preds.get((sid, "sighting"), {}).get("top1_sim", "")
        row["best_pick"], row["best_result"], top5 = pick(uid, "best", true_name)
        row["best_top5"] = "; ".join(top5)
        row["gemini_pick"], row["gemini_result"], _ = pick(uid, "gemini", true_name)
        row["sift_pick"], row["sift_result"], _ = pick(uid, "sift", true_name)
        row["embeddings_pick"], row["embeddings_result"], _ = pick(uid, "combined", true_name)
        row["whole_photo_pick"], row["whole_photo_result"], _ = pick(uid, "full", true_name)
        out.append(row)
    tmp = TIDY.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIDY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    tmp.replace(TIDY)


SKIPPED = ROOT / "results" / "skipped.csv"
META = ROOT / "results" / "photo_meta.csv"  # when each uploaded photo was taken (from EXIF)
_meta_lock = __import__("threading").Lock()


def taken_at(img: Image.Image) -> str:
    """'YYYY-MM-DD HH:MM:SS' from EXIF DateTimeOriginal (or DateTime), '' if absent."""
    try:
        exif = img.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306) or ""
        return raw.replace(":", "-", 2).strip() if raw else ""
    except Exception:
        return ""


def note_meta(upload_id, filename, taken):
    with _meta_lock:
        new = not META.exists()
        META.parent.mkdir(exist_ok=True)
        with META.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["upload_id", "filename", "taken"])
            if new:
                w.writeheader()
            w.writerow({"upload_id": upload_id, "filename": filename, "taken": taken})


def log_skipped(photos):
    """Photos left out of a batch run (e.g. no turtle shell), kept for spot-checking."""
    SKIPPED.parent.mkdir(exist_ok=True)
    new = not SKIPPED.exists()
    with SKIPPED.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "upload_id", "filename", "reason"])
        if new:
            w.writeheader()
        for p in photos:
            w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "upload_id": p["upload_id"],
                        "filename": p["filename"], "reason": p["skipped"]})


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
    try:
        write_tidy()
    except Exception as e:  # never lose an upload over the summary file
        print(f"could not update {TIDY.name}: {e}")


def session_stats():
    """Score each matcher against the recorded true answers."""
    if not LOG.exists():
        return {"uploads": 0}
    rows = list(csv.DictReader(LOG.open()))
    preds = {(r["upload_id"], r["method"]): r for r in rows if r["event"] == "identify"}
    truth = {r["upload_id"]: r for r in rows if r["event"] == "feedback"}  # last wins
    stats = {"uploads": len({u for u, m in preds if m != "sighting"}),
             "scored": len({u for u in truth if not u.startswith("S")}), "bad_photo": 0, "methods": {}}
    for uid, fb in truth.items():
        if fb["verdict"] == "bad_photo":
            stats["bad_photo"] += not uid.startswith("S")
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
    photo_pool = ThreadPoolExecutor(4)  # photos of one sighting run in parallel
    sightings: dict[str, list[str]] = {}  # sighting id -> photo upload ids (restored from the log)
    if LOG.exists():
        for r in csv.DictReader(LOG.open()):
            if r["method"] == "sighting" and r["notes"].startswith("photos "):
                sightings[r["upload_id"]] = r["notes"][len("photos "):].split(";")
    needed = set().union(*(sources_of(m) for m in matchers.values()))
    crop_variants = [v for v in shellcrop.VARIANTS if v in needed]
    classes = sorted(set().union(*(m.classes for m in [*matchers.values(), *([sift] if sift else [])])))
    if sift and "combined" in matchers:
        assert sift.ids == matchers["combined"].parts[0].ids, "SIFT and embedding references differ"
        sift_cidx = np.array([sift.classes.index(n) for n in sift.names])

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    @app.get("/summary", response_class=HTMLResponse)
    def summary_page(sort: str = "time"):
        return summary_html(classes, sort)

    @app.get("/summary.csv")
    def summary_csv():
        write_tidy()
        if not TIDY.exists():
            raise HTTPException(404, "no uploads yet")
        return FileResponse(TIDY, media_type="text/csv", filename="uploads_summary.csv")

    @app.get("/session_log.csv")
    def session_log_csv():
        if not LOG.exists():
            raise HTTPException(404, "no session log yet")
        return FileResponse(LOG, media_type="text/csv", filename="session_log.csv")

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

    def process_photo(raw: bytes, filename: str, upload_id: str, with_gemini: bool = True,
                      require_shell: bool = False):
        """Run every matcher on one photo. Returns its results, log rows, and the
        per-reference evidence (fused + SIFT scores) for combining a sighting."""
        try:
            original = Image.open(io.BytesIO(raw))
            taken = taken_at(original)
            img = ImageOps.exif_transpose(original)
        except Exception:
            raise HTTPException(400, f"could not read {filename}")
        note_meta(upload_id, filename, taken)
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
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
                if require_shell and box is None:  # batch mode: no turtle in this photo
                    return {"upload_id": upload_id, "image": f"/uploads/{path.name}", "skipped": "no turtle shell found",
                            "filename": filename}
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
                                 "filename": filename, "box": box or "",
                                 "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                                 "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                                 "likely_new": int(results["sift"]["likely_new"])})
            except Exception as e:
                results["sift"] = {"error": f"SIFT failed: {str(e)[:250]}"}
        comb = matchers.get("combined")
        evidence = None
        if s_sift is not None and comb and not (sources_of(comb) - vecs.keys()):
            s_emb = comb.sims(vecs)
            fused = matching.fuse(s_sift, s_emb, sift_cidx, len(sift.classes))
            evidence = {"fused": fused, "sift": s_sift}
            results = {"best": best_guess(fused, s_sift, s_emb, images["crop"], box), **results}
            b = results["best"]
            log_rows.insert(0, {"upload_id": upload_id, "event": "identify", "method": "best",
                                "filename": filename, "box": "",
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
                             "filename": filename, "box": r["box"] or "",
                             "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                             "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                             "likely_new": int(r["likely_new"])})
        shortlist = next((results[k] for k in ("best", "combined") if "matches" in results.get(k, {})), None)
        if use_gemini and with_gemini and shortlist and "crop" in images:
            g = gemini_pick(shortlist, images["crop"], upload_id)
            results = {k: v for k, v in [("best", results.get("best")), ("gemini", g),
                                         ("sift", results.get("sift"))] if v} | results
            g = results["gemini"]
            if "matches" in g:
                m = g["matches"]
                log_rows.insert(0, {"upload_id": upload_id, "event": "identify", "method": "gemini",
                                    "filename": filename, "box": g["box"] or "",
                                    "top1": m[0]["name"], "top1_sim": m[0]["sim"],
                                    "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m),
                                    "likely_new": int(g["likely_new"]), "notes": g["reason"]})
        return {"upload_id": upload_id, "image": f"/uploads/{path.name}", "methods": results,
                "log_rows": log_rows, "evidence": evidence}

    @app.post("/api/identify")
    def identify_upload(images: list[UploadFile] = File(...), gemini: str = Form("1"),
                        require_shell: str = Form("0")):
        """One photo, or several photos of the same turtle (one sighting).

        Several photos are matched in parallel and combined by adding up each
        turtle's fused score across them; on the different-day test that took
        top-1 from 0.85 (one photo) to 0.93 (whole sighting).
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-")
        jobs = [(f.file.read(), f.filename, stamp + uuid.uuid4().hex[:6], gemini != "0", require_shell == "1")
                for f in images]
        photos = list(photo_pool.map(lambda a: process_photo(*a), jobs))
        skipped = [p for p in photos if p.get("skipped")]
        if skipped:
            log_skipped(skipped)
        photos = [p for p in photos if not p.get("skipped")]
        if not photos:
            return {"skipped": [{"filename": p["filename"], "reason": p["skipped"], "image": p["image"]} for p in skipped]}
        rows = [r for p in photos for r in p["log_rows"]]
        if len(photos) == 1:
            log(rows)
            return {k: photos[0][k] for k in ("upload_id", "image", "methods")}
        sid = "S" + stamp + uuid.uuid4().hex[:6]
        ids = [p["upload_id"] for p in photos]
        sightings[sid] = ids
        methods = {}
        with_evidence = [p for p in photos if p["evidence"]]
        if with_evidence:
            m = methods["sighting"] = combine_sighting(with_evidence)
            rows.append({"upload_id": sid, "event": "identify", "method": "sighting",
                         "filename": f"{len(with_evidence)} photos", "box": "",
                         "top1": m["matches"][0]["name"], "top1_sim": m["matches"][0]["sim"],
                         "top5": ";".join(f"{x['name']}:{x['sim']}" for x in m["matches"]),
                         "likely_new": int(m["likely_new"]), "notes": "photos " + ";".join(ids)})
        for k, p in enumerate(photos, 1):
            per = p["methods"]
            methods[f"photo{k}"] = per.get("best") or next(iter(per.values()))
        log(rows)
        return {"upload_id": sid, "image": photos[0]["image"],
                "images": [p["image"] for p in photos], "methods": methods}

    def combine_sighting(photos, top=5, per_class=3):
        """Add up each turtle's fused score across photos; refs = best SIFT photos over all."""
        F = sum(p["evidence"]["fused"] for p in photos)
        S = np.max(np.stack([p["evidence"]["sift"] for p in photos]), 0)
        matches = []
        for k in np.argsort(-F)[:top]:
            mine = np.flatnonzero(sift_cidx == k)
            mine = mine[np.argsort(-S[mine])][:per_class]
            matches.append({"name": sift.classes[k], "sim": round(float(F[k]), 2),
                            "refs": [{"path": sift.paths[j], "sim": round(float(S[j]), 1)} for j in mine]})
        spot = round(float(S[sift_cidx == sift.classes.index(matches[0]["name"])].max()), 1)
        return {"image": photos[0]["methods"]["best"]["image"], "box": None, "matches": matches,
                "spot": spot, "likely_new": spot < sift.threshold, "threshold": sift.threshold,
                "n_photos": len(photos)}



    def best_guess(fused, s_sift, s_emb, image_url, box, top=5, per_class=3):
        """Rank turtles by fused SIFT + combined-embedding evidence (matching.fuse)."""
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
        ids = [upload_id, *sightings.get(upload_id, [])]  # a sighting's answer applies to each photo
        log([{"upload_id": u, "event": "feedback", "verdict": verdict, "true_name": true_name,
              "notes": notes} for u in ids])
        return {"stats": session_stats()}

    return app


SUMMARY_METHODS = [("sighting", "All photos combined"), ("best", "SIFT + embeddings"),
                   ("gemini", "Gemini pick"), ("sift", "Spot match (SIFT)"),
                   ("combined", "Embeddings only (crop + tight)"), ("full", "Embeddings (whole photo)"),
                   ("crop", "Embeddings (shell crop)"), ("tight", "Embeddings (tight crop)")]


def band_of(spot):
    """Spot-score band (different-day test: top pick right 98% / ~91% / 65% / 21%)."""
    if spot is None:
        return ""
    return "Confirmed" if spot >= 4 else "Likely" if spot >= 2 else "Possible" if spot >= 1 else "No match"


REVIEW_STYLE = """
:root{--bg:#f6f4ee;--card:#fff;--ink:#1d2a22;--muted:#6b756e;--line:#e2ded3;--ok:#2f6b4f;--mid:#b5651d;--bad:#a33;--hit:#e5f0ea;--grey:#b8b2a4}
@media (prefers-color-scheme: dark){:root{--bg:#141712;--card:#1c201a;--ink:#e6e9de;--muted:#9aa292;--line:#2f352b;--ok:#86b07a;--mid:#e3aa45;--bad:#e07e58;--hit:#22301f;--grey:#5b6257}}
*{box-sizing:border-box}body{margin:0;padding:16px;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:1400px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:24px 0 8px}
.muted{color:var(--muted);font-size:12px}a{color:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;overflow-x:auto;margin-bottom:12px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500}td.ok{color:var(--ok);font-weight:600}td.mid{color:var(--mid)}td.bad{color:var(--bad)}
.thumbs{min-width:190px}.thumbs img{height:84px;border-radius:6px;margin:0 4px 4px 0}
.chip{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600}
.b-Confirmed{background:var(--hit);color:var(--ok)}.b-Likely{background:var(--hit);color:var(--ok)}
.b-Possible{background:#fbeee0;color:var(--mid)}.b-No{background:#f6e3df;color:var(--bad)}
@media (prefers-color-scheme: dark){.b-Possible{background:#3a2e17}.b-No{background:#3a201a}}
.ans{min-width:230px}.ans select{font:inherit;font-size:13px;padding:4px;max-width:150px}
.ans button{font:inherit;font-size:12px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer;margin:2px 2px 0 0}
.ans .now{font-weight:600;margin-bottom:4px}tr.answered{background:color-mix(in srgb,var(--hit) 40%,transparent)}
.big{font-size:20px;font-weight:600}th.sort{cursor:pointer;color:var(--ok);white-space:nowrap}th.sort:hover{text-decoration:underline}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
svg text{fill:var(--muted);font-size:11px}
"""

REVIEW_SCRIPT = """
let sortState = {key: 'taken', dir: 1};
document.addEventListener('click', e => {
  const th = e.target.closest('th.sort'); if(!th) return;
  const key = th.dataset.key;
  sortState = {key, dir: sortState.key === key ? -sortState.dir : (key === 'spot' ? -1 : 1)};
  const tb = document.querySelector('#photos tbody');
  const rows = [...tb.querySelectorAll('tr[data-taken]')];
  const val = r => key === 'spot' ? (r.dataset.spot === '' ? -1 : parseFloat(r.dataset.spot)) : (r.dataset[key] || '');
  rows.sort((a, b) => {
    const x = val(a), y = val(b);
    const c = (typeof x === 'number') ? x - y : x.localeCompare(y, undefined, {numeric: true});
    return c * sortState.dir || a.dataset.taken.localeCompare(b.dataset.taken);
  });
  rows.forEach(r => tb.appendChild(r));
  document.querySelectorAll('th.sort').forEach(h => h.textContent = h.textContent.replace(/ [↑↓↕]$/, '') +
    (h.dataset.key === key ? (sortState.dir > 0 ? ' ↑' : ' ↓') : ' ↕'));
});
document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-verdict]'); if(!b) return;
  const uid = b.dataset.uid, verdict = b.dataset.verdict;
  const name = verdict === 'known' ? document.getElementById('sel-' + uid).value : '';
  const fd = new FormData(); fd.append('upload_id', uid); fd.append('verdict', verdict); fd.append('true_name', name);
  b.disabled = true;
  const r = await fetch('/api/feedback', {method: 'POST', body: fd});
  b.disabled = false;
  const now = document.getElementById('now-' + uid);
  if(!r.ok){ now.textContent = 'Could not save. Try again.'; return; }
  now.textContent = verdict === 'known' ? 'Matches ' + name : verdict.replace('_', ' ');
  document.getElementById('row-' + uid).classList.add('answered');
  document.getElementById('stale').hidden = false;
});
"""


def summary_html(classes=(), sort="time"):
    """Review page: method comparison, new-turtle threshold analysis, and every photo with
    its picks and tap-to-confirm controls."""
    esc = lambda t: html.escape(str(t), quote=True)
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–"
    rows = list(csv.DictReader(LOG.open())) if LOG.exists() else []
    preds = {(r["upload_id"], r["method"]): r for r in rows if r["event"] == "identify"}
    truth = {r["upload_id"]: r for r in rows if r["event"] == "feedback"}  # last wins
    sets = {r["upload_id"]: r["notes"][len("photos "):].split(";") for r in rows
            if r["event"] == "identify" and r["method"] == "sighting"}
    set_of = {pid: sid for sid, pids in sets.items() for pid in pids}
    top5 = lambda uid, m: [x.rsplit(":", 1)[0] for x in preds[(uid, m)]["top5"].split(";")] if (uid, m) in preds else []

    def spot_of(uid):
        b = preds.get((uid, "best"), {})
        try:
            return float(b.get("notes", "").replace("spot ", "")) if b.get("notes", "").startswith("spot") \
                else float(preds[(uid, "sift")]["top1_sim"])
        except (KeyError, ValueError):
            return None

    taken = {}
    if META.exists():
        taken = {r["upload_id"]: r["taken"] for r in csv.DictReader(META.open())}
    recs = []
    for uid in {r["upload_id"] for r in rows if r["event"] == "identify" and r["method"] != "sighting"}:
        fb = truth.get(uid) or truth.get(set_of.get(uid, ""))
        verdict = fb["verdict"] if fb else ""
        true_name = fb["true_name"] if verdict == "known" else None
        any_row = next(r for (u, _), r in preds.items() if u == uid)
        best = top5(uid, "best")
        recs.append({"uid": uid, "file": any_row["filename"], "time": any_row["time"], "spot": spot_of(uid),
                     "taken": taken.get(uid, ""),
                     "verdict": verdict, "true": true_name, "best": best[0] if best else "",
                     "right": bool(true_name and best and best[0] == true_name)})
    recs.sort(key=lambda r: r["taken"] or r["time"])

    # --- method comparison
    m = session_stats().get("methods", {})
    method_rows = "".join(
        f"<tr><td>{label}</td><td>{pct(x['top1'], x['known'])}</td><td>{pct(x['top5'], x['known'])}</td>"
        f"<td>{pct(x['new_flagged'], x['new'])}</td></tr>"
        for key, label in SUMMARY_METHODS if (x := m.get(key)))

    # --- threshold analysis on confirmed photos (spot score of the SIFT + embeddings pick)
    known = [r for r in recs if r["verdict"] == "known" and r["spot"] is not None]
    new = [r for r in recs if r["verdict"] == "new_turtle" and r["spot"] is not None]
    thr_rows = ""
    for t in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5):
        k_above = [r for r in known if r["spot"] >= t]
        thr_rows += (f"<tr><td>{t:g}</td><td>{pct(len(k_above), len(known))}</td>"
                     f"<td>{pct(sum(r['right'] for r in k_above), len(k_above))}</td>"
                     f"<td>{pct(sum(r['spot'] < t for r in new), len(new))}</td></tr>")
    band_rows = ""
    for band in ("Confirmed", "Likely", "Possible", "No match"):
        inb = [r for r in recs if band_of(r["spot"]) == band]
        kb = [r for r in inb if r["verdict"] == "known"]
        band_rows += (f"<tr><td><span class='chip b-{band.split()[0]}'>{band}</span></td><td>{len(inb)}</td>"
                      f"<td>{pct(sum(r['right'] for r in kb), len(kb))}</td>"
                      f"<td>{sum(r['verdict'] == 'new_turtle' for r in inb)}</td>"
                      f"<td>{sum(not r['verdict'] for r in inb)}</td></tr>")

    # strip chart: every photo's spot score, by what it turned out to be
    W, H, X0, XMAX = 900, 150, 130, 12.0
    xs = lambda v: X0 + min(v, XMAX) / XMAX * (W - X0 - 20)
    lanes = [("known, top pick right", lambda r: r["verdict"] == "known" and r["right"], "var(--ok)"),
             ("known, top pick wrong", lambda r: r["verdict"] == "known" and not r["right"], "var(--mid)"),
             ("new turtle", lambda r: r["verdict"] == "new_turtle", "var(--bad)"),
             ("not confirmed yet", lambda r: not r["verdict"], "var(--grey)")]
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Spot scores by outcome">']
    for v in (1, 2, 4):
        svg.append(f'<line x1="{xs(v)}" x2="{xs(v)}" y1="8" y2="{H - 22}" stroke="var(--line)" stroke-dasharray="3 3"/>'
                   f'<text x="{xs(v) + 3}" y="16">{v}</text>')
    for v in (0, 6, 8, 10, 12):
        svg.append(f'<text x="{xs(v) - 4}" y="{H - 6}">{v}{"+" if v == 12 else ""}</text>')
    for i, (name, test, color) in enumerate(lanes):
        y = 30 + i * 28
        svg.append(f'<text x="4" y="{y + 4}">{name}</text>')
        for j, r in enumerate([r for r in recs if test(r) and r["spot"] is not None]):
            svg.append(f'<circle cx="{xs(r["spot"]):.1f}" cy="{y + (j % 3 - 1) * 5}" r="4.5" fill="{color}" opacity=".8">'
                       f'<title>{esc(r["file"])}: spot {r["spot"]}</title></circle>')
    svg.append("</svg>")

    # --- photo rows
    opts = lambda sel: "".join(f'<option{" selected" if c == sel else ""}>{esc(c)}</option>' for c in classes)
    cols = [("best", "SIFT + emb."), ("sift", "SIFT only"), ("combined", "Emb. only"),
            ("tight", "Emb. tight"), ("full", "Emb. whole"), ("gemini", "Gemini")]

    def cell(uid, method, true_name):
        t = top5(uid, method)
        if not t:
            return "<td class='muted'>–</td>"
        mark, cls = "", ""
        if true_name:
            mark, cls = ((" ✓", "ok") if t[0] == true_name else (" (top 5)", "mid") if true_name in t else (" ✗", "bad"))
        return f"<td class='{cls}'>{esc(t[0])}{mark}</td>"

    body = []
    for r in recs:
        uid = r["uid"]
        crop = UPLOADS / f"{uid}_crop.jpg"
        thumbs = (f'<a href="/uploads/{uid}.jpg" target="_blank"><img src="/uploads/{uid}.jpg" loading="lazy" alt=""></a>'
                  + (f'<a href="/uploads/{uid}_crop.jpg" target="_blank"><img src="/uploads/{uid}_crop.jpg" loading="lazy" alt=""></a>'
                     if crop.exists() else ""))
        now = ("Matches " + esc(r["true"])) if r["verdict"] == "known" else esc(r["verdict"].replace("_", " ")) or \
              "<span class='muted'>not confirmed</span>"
        band = band_of(r["spot"])
        turtle = r["true"] if r["verdict"] == "known" else (r["best"] if not r["verdict"] else r["verdict"].replace("_", " "))
        body.append(
            f"<tr id='row-{uid}' class='{'answered' if r['verdict'] else ''}' data-taken='{esc(r['taken'] or r['time'].replace('T', ' '))}'"
            f" data-turtle='{esc(turtle.lower())}' data-spot='{'' if r['spot'] is None else r['spot']}' data-file='{esc(r['file'].lower())}'>"
            f"<td class='thumbs'>{thumbs}</td>"
            f"<td><b>{esc(r['taken'][:10]) or '–'}</b><br>{esc(r['taken'][11:16])}"
            f"<div class='muted'>{'' if r['taken'] else 'uploaded ' + esc(r['time'][11:16])}</div></td>"
            f"<td><b>{esc(turtle)}</b><div class='muted'>{'confirmed' if r['verdict'] == 'known' else ('your answer' if r['verdict'] else 'best guess')}</div></td>"
            f"<td class='muted'>{esc(r['file'])}{'<br>set ' + esc(set_of[uid][-6:]) if uid in set_of else ''}</td>"
            f"<td>{'' if r['spot'] is None else r['spot']}<br><span class='chip b-{band.split()[0] if band else ''}'>{band}</span></td>"
            f"<td class='ans'><div class='now' id='now-{uid}'>{now}</div>"
            f"<select id='sel-{uid}' aria-label='Turtle'>{opts(r['true'] or r['best'])}</select>"
            f"<button data-uid='{uid}' data-verdict='known'>Matches</button><br>"
            f"<button data-uid='{uid}' data-verdict='new_turtle'>New turtle</button>"
            f"<button data-uid='{uid}' data-verdict='bad_photo'>Bad photo</button></td>"
            + "".join(cell(uid, mth, r["true"]) for mth, _ in cols) + "</tr>")

    skipped = list(csv.DictReader(SKIPPED.open())) if SKIPPED.exists() else []
    no_turtle = [s for s in skipped if s["upload_id"]]
    dupes = [s for s in skipped if not s["upload_id"]]
    skipped_html = "".join(
        f'<figure style="margin:0"><a href="/uploads/{esc(s["upload_id"])}.jpg" target="_blank">'
        f'<img src="/uploads/{esc(s["upload_id"])}.jpg" loading="lazy" style="height:90px;border-radius:6px" alt=""></a>'
        f'<figcaption class="muted">{esc(s["filename"])}</figcaption></figure>' for s in no_turtle)
    dupes_html = ", ".join(f"{esc(s['filename'])} ({esc(s['reason'])})" for s in dupes)

    n_conf = len([r for r in recs if r["verdict"]])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Turtle ID Review</title>
<style>{REVIEW_STYLE}</style></head><body><main>
<h1>Turtle ID review</h1>
<div class="muted">{len(recs)} photos matched · {n_conf} confirmed · {len(no_turtle)} skipped (no turtle) · {len(dupes)} duplicates skipped ·
<a href="/summary.csv">summary CSV</a> · <a href="/session_log.csv">raw log</a> · <a href="/">back to the app</a></div>
<p id="stale" hidden class="muted" style="font-size:14px">Answers saved. <a href="">Refresh</a> to update the numbers.</p>

<h2>Where is the "new turtle" line?</h2>
<div class="card">{"".join(svg)}
<p class="muted">Each dot is a photo, placed by its spot score (SIFT + embeddings pick; 12+ shown at 12). Hover for the file name. Confirm photos below to sort them into the right lane.</p></div>
<div class="grid">
<div class="card"><b>If we call anything below the cut-off a new turtle…</b>
<table><tr><th>cut-off</th><th>known turtles kept</th><th>…and top pick right</th><th>new turtles caught</th></tr>{thr_rows}</table>
<p class="muted">Confirmed photos only: {len(known)} known, {len(new)} new.</p></div>
<div class="card"><b>By band</b>
<table><tr><th>band</th><th>photos</th><th>known: top pick right</th><th>new</th><th>unconfirmed</th></tr>{band_rows}</table>
<p class="muted">On the earlier different-day test the top pick was right 98% (Confirmed, 4+), ~91% (Likely, 2–4), 65% (Possible, 1–2) and 21% (No match, under 1) of the time.</p></div>
</div>

<h2>How each method did</h2>
<div class="card"><table><tr><th>method</th><th>right first time</th><th>right turtle in top 5</th><th>new turtles flagged (spot &lt; 4 / weak)</th></tr>
{method_rows or '<tr><td colspan=4 class=muted>Confirm some photos below to score the methods.</td></tr>'}</table></div>

<h2>Every photo</h2>
<div class="muted" style="margin-bottom:6px">One photo per row. <b>Click a column header to sort</b> (again to reverse); sorting by turtle groups each turtle's photos.
Pick the turtle and tap <b>Matches</b>, or tap <b>New turtle</b> / <b>Bad photo</b>. Click a photo to open it full size.</div>
<div class="card"><table id="photos"><thead><tr><th>photo · shell crop</th>
<th class="sort" data-key="taken">taken ↕</th><th class="sort" data-key="turtle">turtle ↕</th><th class="sort" data-key="file">file ↕</th>
<th class="sort" data-key="spot">spot ↕</th><th>your answer</th>{"".join(f"<th>{h}</th>" for _, h in cols)}</tr></thead>
<tbody>{"".join(body) or '<tr><td colspan=12 class=muted>No photos yet.</td></tr>'}</tbody></table>
<p class="muted">✓ right first time · (top 5) in the top 5 · ✗ not in the top 5.</p></div>

<h2>Skipped: no turtle shell found</h2>
<div class="card" style="display:flex;flex-wrap:wrap;gap:8px">{skipped_html or '<span class=muted>None.</span>'}</div>
<h2>Skipped: duplicates</h2>
<div class="card muted" style="font-size:13px">{dupes_html or 'None.'}</div>
</main><script>{REVIEW_SCRIPT}</script></body></html>"""
    return page


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
label.drop.over{border-color:var(--accent);background:var(--hit);color:var(--ink)}
label.drop.busy::after{content:"Working on a set: drop more to queue them";display:block;margin-top:6px;font-size:13px;color:var(--accent)}
.working .status{display:flex;align-items:center;gap:10px;margin:10px 0 8px;font-weight:600}
.spin{width:16px;height:16px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.bar{height:6px;background:var(--line);border-radius:99px;overflow:hidden;margin-bottom:8px}
.bar span{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
.bar span.indet{width:100%!important;opacity:.55;animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.25}}
.tile{width:160px;height:120px;border-radius:8px;background:var(--hit);color:var(--muted);display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px;padding:8px;overflow-wrap:anywhere}
@media (prefers-reduced-motion: reduce){.spin,.bar span.indet{animation:none}}
.query{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}.qimgs{display:flex;gap:6px;flex-wrap:wrap}.qimgs img{width:160px;max-width:100%;border-radius:8px}
.cols{display:flex;gap:14px;overflow-x:auto;scroll-snap-type:x mandatory;scroll-padding-left:2px;padding-bottom:10px;align-items:flex-start}
.cols>.col{flex:0 0 min(340px,86vw);scroll-snap-align:start;margin-bottom:0}
.colnav{display:flex;align-items:center;gap:8px;margin:0 0 8px}.colnav .muted{flex:1}
.desc{font-size:13px;color:var(--muted);background:var(--bg);border-radius:8px;padding:8px 10px;margin:0 0 10px;line-height:1.4}.desc b{color:var(--ink);font-weight:600}
.col h2{font-size:16px;margin:0 0 8px}.cropimg{display:block;margin-bottom:8px}.cropimg img{max-height:180px;max-width:100%;border-radius:8px}
.banner{padding:8px 10px;border-radius:8px;font-weight:600;font-size:14px;margin:6px 0}
.banner.new{background:#fbeee0;color:var(--warn)}.banner.known{background:var(--hit);color:var(--ok)}
.match{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center;border-top:1px solid var(--line);padding:8px 0}
.match.truth{background:var(--hit);margin:0 -8px;padding:8px}
.match .name{font-weight:600}.match .sim{color:var(--muted);font-size:13px}
.refs{grid-column:1/-1;display:flex;gap:6px;overflow-x:auto}.refs img{height:84px;border-radius:6px}
button{font:inherit;font-size:14px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 10px;cursor:pointer}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}select,input[type=text]{font:inherit;padding:7px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;font-size:14px;width:100%}
table.glance td{vertical-align:top}tr.go{cursor:pointer}tr.go:hover{background:var(--hit)}
.okc{color:var(--ok);font-weight:600}.warn{color:var(--warn)}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.done{color:var(--ok);font-weight:600}.err{color:var(--bad)}.muted{color:var(--muted);font-size:13px}
</style></head><body><main>
<h1>Turtle ID</h1>
<div class="sub"><span id="meta">loading…</span> · <a href="/summary" target="_blank">session summary</a> · <a href="/crops" target="_blank">review reference crops</a></div>
<div class="card">
  <label class="drop" id="dropbox"><input id="file" type="file" accept="image/*" multiple><b>Choose, take or drop photos of one turtle</b><br>top, left and right of the shell, filling the frame. Several photos are combined.</label>
</div>
<div id="result"></div>
<div class="card" id="stats"></div>
</main><script>
let INFO, CUR, TRUTH;
const LABEL = {sighting: 'All photos combined (best)', best: 'SIFT + embeddings (best)', gemini: 'Gemini pick (second opinion)', sift: 'Spot match (SIFT)', combined: 'Embeddings only (crop + tight)', full: 'Whole photo', crop: 'Cropped to shell', tight: 'Tight (inside shell)'};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const label = m => LABEL[m] || (m.startsWith('photo') ? `Photo ${m.slice(5)} alone` : m);
const pct = (a, b) => b ? `${a}/${b} (${Math.round(100*a/b)}%)` : '–';
function showStats(s){
  if(!s || !s.uploads){ $('stats').innerHTML = '<span class="muted">No uploads yet this session.</span>'; return; }
  const rows = Object.entries(s.methods || {}).map(([m, x]) => `<tr><td>${label(m)}</td>
    <td>${pct(x.top1, x.known)}</td><td>${pct(x.top5, x.known)}</td>
    <td>${pct(x.new_flagged, x.new)}</td><td>${pct(x.known_flagged_new, x.known)}</td></tr>`).join('');
  $('stats').innerHTML = `<div class="muted" style="margin-bottom:6px">Session: ${s.uploads} uploaded, ${s.scored} scored${s.bad_photo ? `, ${s.bad_photo} bad photos` : ''}</div>
    <table><tr><th>matcher</th><th>top-1 right</th><th>in top 5</th><th>new turtles flagged weak</th><th>known flagged weak</th></tr>${rows}</table>`;
}
async function load(){
  INFO = await (await fetch('/api/info')).json();
  $('meta').textContent = `${INFO.classes.length} known turtles · ` + Object.entries(INFO.methods).map(([m, x]) =>
    `${label(m)}: ${x.n_ref} refs, "new" below ${x.threshold}`).join(' · ');
  showStats(INFO.stats);
}
let BUSY = false;
const QUEUE = [];
const isImage = f => f.type.startsWith('image/') || /[.](heic|heif)$/i.test(f.name);
const canPreview = f => /^image[/](jpeg|png|gif|webp)$/.test(f.type);
function upload(files){
  files = files.filter(isImage);
  if(!files.length) return;
  QUEUE.push(files);
  if(BUSY) showQueue(); else next();
}
function showQueue(){
  const el = $('queued');
  if(el) el.textContent = QUEUE.length ? `${QUEUE.length} more ${QUEUE.length > 1 ? 'sets' : 'set'} waiting` : '';
}
function post(files, onProgress){
  return new Promise((resolve, reject) => {
    const fd = new FormData(); files.forEach(f => fd.append('images', f));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/identify');
    xhr.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.upload.onload = () => onProgress(1);
    xhr.onload = () => {
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch(_) {}
      xhr.status < 300 ? resolve(body) : reject(new Error(body.detail || `Upload failed (${xhr.status}). Try again.`));
    };
    xhr.onerror = () => reject(new Error('Could not reach the app. Is it still running?'));
    xhr.send(fd);
  });
}
async function next(){
  const files = QUEUE.shift();
  if(!files){ BUSY = false; $('dropbox').classList.remove('busy'); return; }
  BUSY = true; $('dropbox').classList.add('busy');
  const urls = files.map(f => canPreview(f) ? URL.createObjectURL(f) : null);
  const tiles = files.map((f, i) => urls[i] ? `<img src="${urls[i]}" alt="">`
    : `<div class="tile">${esc(f.name)}</div>`).join('');
  const n = files.length, start = Date.now();
  $('result').innerHTML = `<div class="card working"><div class="qimgs">${tiles}</div>
    <div class="status"><span class="spin" aria-hidden="true"></span><span id="stage">Uploading ${n > 1 ? n + ' photos' : 'photo'}…</span></div>
    <div class="bar"><span id="barfill"></span></div>
    <div class="muted" id="hint">Usually 5–10 seconds per set${n > 1 ? '; photos are matched in parallel' : ''}.</div>
    <div class="muted" id="queued"></div></div>`;
  showQueue();
  let stage = 'upload';
  const tick = setInterval(() => {
    if(stage === 'match') $('stage').textContent = `Cropping and matching ${n > 1 ? n + ' photos' : ''}… ${Math.round((Date.now() - start) / 1000)} s`;
  }, 500);
  let shown = false;
  try {
    CUR = await post(files, frac => {
      $('barfill').style.width = `${Math.round(frac * 100)}%`;
      if(frac >= 1 && stage === 'upload'){ stage = 'match'; $('barfill').classList.add('indet'); }
    });
    TRUTH = null; render(); shown = true;
    if(QUEUE.length) $('saved').insertAdjacentHTML('beforebegin', `<p class="muted">${QUEUE.length} more ${QUEUE.length > 1 ? 'sets' : 'set'} waiting. Record this one, then the next will show.</p>`);
  } catch(err){
    $('result').innerHTML = `<div class="card err">${esc(err.message)}</div>`;
  } finally {
    clearInterval(tick); urls.forEach(u => u && URL.revokeObjectURL(u));
    if(QUEUE.length && shown) await waitForVerdictOrTimeout();
    next();
  }
}
// With a queue, keep each result on screen until it's recorded (or 60 s pass).
let verdictResolve = null;
function waitForVerdictOrTimeout(){
  return new Promise(res => { verdictResolve = res; setTimeout(() => { verdictResolve = null; res(); }, 60000); });
}
$('file').onchange = e => { const files = [...e.target.files]; e.target.value = ''; upload(files); };
// Drag and drop anywhere on the page; several photos dropped together = one turtle.
let dragDepth = 0;
const drop = document.querySelector('label.drop');
addEventListener('dragenter', e => { if([...e.dataTransfer.types].includes('Files')){ dragDepth++; drop.classList.add('over'); } });
addEventListener('dragleave', () => { if(--dragDepth <= 0){ dragDepth = 0; drop.classList.remove('over'); } });
addEventListener('dragover', e => { if([...e.dataTransfer.types].includes('Files')) e.preventDefault(); });
addEventListener('drop', e => {
  if(![...e.dataTransfer.types].includes('Files')) return;
  e.preventDefault(); dragDepth = 0; drop.classList.remove('over');
  upload([...e.dataTransfer.files]);
});
// Spot-score bands, from the different-day test (how often the top pick was right):
// 4+ 98%, 2-4 ~91%, 1-2 65%, under 1 21%.
function spotBanner(s, name){
  const n = esc(name);
  if(s >= 4) return `<div class="banner known">Confirmed by spot match: ${n} (score ${s})</div>`;
  if(s >= 2) return `<div class="banner known">Likely match: ${n} (score ${s}; ~91% right in testing)</div>`;
  if(s >= 1) return `<div class="banner new">Possible match: ${n} (score ${s}; ~65% right in testing). Check by eye</div>`;
  return `<div class="banner new">No spot match found (score ${s}): could be a new turtle, or a view we don't have</div>`;
}
function column(method, r){
  if(r.error) return `<div class="card col" id="col-${method}"><h2>${label(method)}</h2><p class="err">${esc(r.error)}</p></div>`;
  const m = r.matches;
  const spot = r.spot ?? m[0].sim;
  const banner = (method === 'sift' || r.spot !== undefined)
    ? spotBanner(spot, m[0].name)
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
  const [uses, how, tested] = descOf(method);
  const desc = uses ? `<div class="desc"><b>${esc(uses)}</b><br>${esc(how)}${tested ? `<br><i>${esc(tested)}</i>` : ''}</div>` : '';
  return `<div class="card col" id="col-${method}"><h2>${label(method)}${method !== 'full' && !r.box ? ' <span class="muted">(no shell found)</span>' : ''}</h2>${desc}${thumb}${banner}${reason}${rows}</div>`;
}
// What each method is, in plain words. Tested = the different-day test (534 photos matched only
// against photos of the same turtle from other days): right first time / right turtle in the top 5.
const DESC = {
  sighting: ['Uses: SIFT + embeddings, every photo in this set',
             'Each turtle\u2019s SIFT + embeddings score is added up across all the photos you uploaded together. Spot score = the best SIFT score any of the photos got for that turtle.',
             'Tested: 93% first time (97% with 4+ photos)'],
  best:     ['Uses: SIFT + embeddings, shell crop',
             'Your shell crop is compared with all 628 reference photos two ways: SIFT spot matching and image embeddings. The two scores are added per turtle. Spot score 4+ = confirmed.',
             'Tested: 85% first time, 92% in top 5'],
  gemini:   ['Uses: Gemini (AI vision), choosing from the SIFT + embeddings top 5',
             'Gemini sees your photo and 3 reference photos of each of those 5 turtles and picks the same shell pattern. It cannot bring in a turtle outside those 5, and it always picks someone, even for a new turtle.',
             'Tested: 86% first time'],
  sift:     ['Uses: SIFT only (box-turtle-id\u2019s matcher), shell crop',
             'Finds small distinctive spots on your shell crop and counts how many have a twin in each reference shell crop. Score = % of spots matched. 4+ = confirmed; under 2 is noise.',
             'Tested: 82% first time, 88% in top 5'],
  combined: ['Uses: embeddings only (no SIFT), shell crop + tight crop',
             'Google image embeddings of both crops, tuned with turtle names (LDA) and averaged. Number = similarity, 0 to 1. Judges overall look, not specific spots.',
             'Tested: 55% first time, 84% in top 5'],
  full:     ['Uses: embeddings only (no SIFT), whole photo',
             'Google image embedding of the uncropped photo, background and hands included. Number = similarity, 0 to 1.',
             'Tested: 39% first time, 68% in top 5'],
  crop:     ['Uses: embeddings only (no SIFT), shell crop',
             'Google image embedding of the shell crop (Gemini\u2019s box + 5% margin). Number = similarity, 0 to 1.',
             'Tested: 40% first time, 68% in top 5'],
  tight:    ['Uses: embeddings only (no SIFT), tight crop',
             'Google image embedding of the centre 71% of the shell box: all shell, edge plates cut off. Number = similarity, 0 to 1.',
             'Tested: 44% first time, 74% in top 5'],
};
const descOf = m => DESC[m] || (m.startsWith('photo')
  ? ['Uses: SIFT + embeddings, this photo alone', 'The same method as the SIFT + embeddings column, run on just this one photo.', ''] : ['', '', '']);
function glance(){
  const rows = Object.entries(CUR.methods).map(([m, r]) => {
    if(r.error) return `<tr><td>${label(m)}</td><td colspan="3" class="err">failed</td></tr>`;
    const top = r.matches[0], spot = r.spot ?? (m === 'sift' ? top.sim : null);
    const state = spot != null
      ? (r.likely_new ? `<span class="warn">spot ${spot}, not confirmed</span>` : `<span class="okc">spot ${spot}, confirmed</span>`)
      : `<span class="muted">similarity ${top.sim}</span>`;
    const hit = TRUTH ? (top.name === TRUTH ? ' ✓' : (r.matches.some(x => x.name === TRUTH) ? ' (top 5)' : ' ✗')) : '';
    const [uses, , tested] = descOf(m);
    return `<tr class="go" data-col="col-${m}"><td>${label(m)}<div class="muted">${esc(uses)}${tested ? '<br>' + esc(tested) : ''}</div></td>
      <td><b>${esc(top.name)}</b>${hit}</td><td>${state}</td><td class="muted">${r.matches.slice(1).map(x => esc(x.name)).join(', ')}</td></tr>`;
  }).join('');
  return `<div class="card"><b>At a glance</b> <span class="muted">· click a row to jump to its details</span>
    <div style="overflow-x:auto"><table class="glance"><tr><th>method</th><th>first pick</th><th>evidence</th><th>rest of top 5</th></tr>${rows}</table></div></div>`;
}
document.addEventListener('click', e => {
  const tr = e.target.closest('tr.go'); if(!tr) return;
  document.getElementById(tr.dataset.col)?.scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'start'});
});
function slide(dir){
  const c = $('cols'); if(!c) return;
  const w = (c.querySelector('.col')?.offsetWidth || 340) + 14;
  c.scrollBy({left: dir * w, behavior: 'smooth'});
}
function render(){
  const opts = INFO.classes.map(c => `<option>${esc(c)}</option>`).join('');
  const queryImgs = (CUR.images || [CUR.image]).map(u => `<img src="${u}">`).join('');
  $('result').innerHTML = `<div class="card"><div class="query"><div class="qimgs">${queryImgs}</div>
      <div style="flex:1;min-width:240px"><b>${CUR.images ? 'Which turtle are these photos of?' : 'What is it really?'}</b>
        <div class="muted">Tap "This is it" on the right turtle below, or:</div>
        <div class="row"><select id="truename"><option value="">Not in either list — pick…</option>${opts}</select>
          <button onclick="send('known', $('truename').value)">Save</button></div>
        <div class="row"><button onclick="send('new_turtle')">New turtle</button><button onclick="send('bad_photo')">Bad photo</button></div>
        <div class="row"><input type="text" id="notes" placeholder="notes (optional)" style="flex:1"></div>
        <div id="saved"></div></div></div></div>
    ${glance()}
    <div class="colnav"><span class="muted">${Object.keys(CUR.methods).length} methods side by side: scroll sideways or use the arrows</span>
      <button onclick="slide(-1)" aria-label="Previous method">←</button><button onclick="slide(1)" aria-label="Next method">→</button></div>
    <div class="cols" id="cols">${Object.entries(CUR.methods).map(([m, r]) => column(m, r)).join('')}</div>`;
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
  if(verdictResolve){ const go = verdictResolve; verdictResolve = null; setTimeout(go, 1200); }
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
