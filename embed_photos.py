"""Embed every photo under data/<name>/*.jpg with gemini-embedding-2-preview (3072 dims)
and store float32 vectors in data/embeddings.sqlite. Resume-safe: re-running only
processes images that don't yet have an embedding for the configured model."""
import os, sys, sqlite3, time, hashlib, mimetypes, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import struct

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "embeddings.sqlite"
ENV = ROOT / ".env"
MODEL = "gemini-embedding-2-preview"
DIM = 3072
WORKERS = 4
RETRIES = 6

# Load .env
for line in ENV.read_text().splitlines():
    if line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1); os.environ.setdefault(k, v)

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

def make_client():
    """Vertex AI (ADC auth) if GOOGLE_GENAI_USE_VERTEXAI is set, else Gemini API key."""
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        return genai.Client(vertexai=True,
                            project=os.environ["GOOGLE_CLOUD_PROJECT"],
                            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


client = make_client()

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    capture_id   INTEGER PRIMARY KEY,
    turtle_name  TEXT NOT NULL,        -- folder name under data/
    file_path    TEXT NOT NULL UNIQUE,
    bytes        INTEGER NOT NULL,
    sha1         TEXT NOT NULL,
    model        TEXT,
    dim          INTEGER,
    embedded_at  TEXT,
    embedding    BLOB                  -- float32 little-endian, len = dim*4
);
CREATE INDEX IF NOT EXISTS idx_images_turtle ON images(turtle_name);
CREATE INDEX IF NOT EXISTS idx_images_model  ON images(model);
"""

def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.commit()
    return con

def discover():
    rows = []
    for jpg in sorted(DATA.glob("*/*.jpg")):
        try:
            cid = int(jpg.stem)
        except ValueError:
            continue
        rows.append((cid, jpg.parent.name, str(jpg.relative_to(ROOT))))
    return rows

def upsert_image_meta(con, cid, name, rel_path, raw):
    sha = hashlib.sha1(raw).hexdigest()
    con.execute(
        """INSERT INTO images(capture_id, turtle_name, file_path, bytes, sha1)
           VALUES(?,?,?,?,?)
           ON CONFLICT(capture_id) DO UPDATE SET
             turtle_name=excluded.turtle_name,
             file_path=excluded.file_path,
             bytes=excluded.bytes,
             sha1=excluded.sha1
           WHERE images.sha1 != excluded.sha1""",
        (cid, name, rel_path, len(raw), sha),
    )

def already_done_ids(con):
    return {r[0] for r in con.execute(
        "SELECT capture_id FROM images WHERE model=? AND dim=? AND embedding IS NOT NULL",
        (MODEL, DIM)
    ).fetchall()}

def embed_one(cid, raw, mt):
    last = None
    for attempt in range(RETRIES):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=raw, mime_type=mt)],
                config=types.EmbedContentConfig(output_dimensionality=DIM),
            )
            vals = result.embeddings[0].values
            if len(vals) != DIM:
                raise ValueError(f"unexpected dim {len(vals)}")
            return struct.pack(f"<{DIM}f", *vals)
        except genai_errors.APIError as e:
            last = e
            code = getattr(e, "code", None) or getattr(getattr(e,"response",None),"status_code",None)
            # 429 / 5xx → backoff; others bail
            if code in (429, 500, 502, 503, 504) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(min(60.0, 1.5 * (2 ** attempt)))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(min(30.0, 1.5 * (2 ** attempt)))
    raise last

write_lock = threading.Lock()

def worker(cid, name, rel_path):
    p = ROOT / rel_path
    raw = p.read_bytes()
    mt = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    blob = embed_one(cid, raw, mt)
    return cid, name, rel_path, raw, blob

def main():
    con = init_db()
    all_rows = discover()
    # Ensure metadata exists for all images (cheap; reads file once)
    print(f"discovered {len(all_rows)} images; ensuring metadata…")
    for cid, name, rel in all_rows:
        try:
            raw = (ROOT / rel).read_bytes()
        except FileNotFoundError:
            continue
        with write_lock:
            upsert_image_meta(con, cid, name, rel, raw)
    con.commit()

    done = already_done_ids(con)
    todo = [r for r in all_rows if r[0] not in done]
    print(f"already embedded: {len(done)};  to embed: {len(todo)};  model={MODEL} dim={DIM}")
    if not todo:
        print("nothing to do.")
        return

    ok = fail = 0
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(worker, *r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            cid, name, rel = futs[fut]
            try:
                cid, name, rel, raw, blob = fut.result()
                with write_lock:
                    con.execute(
                        """UPDATE images SET model=?, dim=?, embedded_at=datetime('now'), embedding=?
                           WHERE capture_id=?""",
                        (MODEL, DIM, blob, cid),
                    )
                    if i % 25 == 0:
                        con.commit()
                ok += 1
            except Exception as e:
                fail += 1
                failures.append((cid, name, type(e).__name__, str(e)[:200]))
            if i % 25 == 0 or i == len(todo):
                rate = i / max(1e-3, time.time() - t0)
                eta = (len(todo) - i) / max(1e-3, rate)
                print(f"  [{i}/{len(todo)}] ok={ok} fail={fail}  {rate:.1f}/s  eta {eta:.0f}s",
                      flush=True)
        con.commit()

    print(f"\nDONE  ok={ok}  fail={fail}  elapsed={time.time()-t0:.1f}s")
    if failures:
        print("failures (first 15):")
        for f in failures[:15]:
            print(" ", f)
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main() or 0)
