"""Build a local review page from label_audit.py's flags (one self-contained HTML file).

Each flagged photo is shown next to its evidence, labelled the way Airtable shows
it (turtle, date, time, view, original filename). Verdicts and notes are kept in
the browser and can be downloaded as a CSV. Nothing is uploaded anywhere.

    python analysis/audit/review_page.py [out.html]   # default ~/Desktop/turtle-review/audit.html
"""
import base64
import csv
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

TV = Path(__file__).resolve().parents[2]
OUT_DIR = TV / "analysis/out/audit"
VIEWS = {"carapace_top": "top", "carapace_left": "left side", "carapace_right": "right side", "front": "front",
         "back": "back", "rear": "rear", "plastron": "bottom (plastron)", "site_view": "site view"}
# confirmed by the user on 2026-10-11
CONFIRMED = {"718": "You confirmed: Jigsaw", "719": "You confirmed: Jigsaw", "720": "You confirmed: Jigsaw",
             "721": "You confirmed: Jigsaw", "201": "You confirmed: Flower", "1015": "You confirmed: Flower"}

meta = {r["capture_id"]: r for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
# where each photo lives in box-turtle-id (from its public API): a turtle's own record or an encounter
_api = OUT_DIR / "captures_api.json"
api = json.loads(_api.read_text()) if _api.exists() else {}
FIXED = {"718": "Jigsaw", "719": "Jigsaw", "720": "Jigsaw", "721": "Jigsaw", "201": "Flower"}  # confirmed moves
orig = {p.stem: p for p in (TV / "data").glob("*/*.jpg") if p.parent.name not in ("crops", "crops_tight")}
esc = lambda s: html.escape(str(s), quote=True)


def when(cid):
    """'June 16, 2025 · 9:31:19 pm' from the original filename, else the date."""
    fn = meta.get(cid, {}).get("original_filename", "")
    try:
        t = datetime.strptime(fn[:19], "%Y-%m-%d %H.%M.%S")
        return t.strftime("%B %-d, %Y · %-I:%M:%S %p").replace("AM", "am").replace("PM", "pm")
    except ValueError:
        d = meta.get(cid, {}).get("captured_date", "")
        return datetime.strptime(d, "%Y-%m-%d").strftime("%B %-d, %Y") if d else "date unknown"


def thumb(cid, side=340):
    im = ImageOps.exif_transpose(Image.open(orig[cid])).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def source(cid):
    a = api.get(cid)
    if not a:
        return ""
    return "on the turtle's own record" if a.get("encounter_id") is None else "in an encounter"


def card(cid, role, note=""):
    m = meta.get(cid, {})
    return f"""<figure class="ph"><img src="{thumb(cid)}" alt="{esc(role)}" loading="lazy">
<figcaption><span class="role">{esc(role)}</span><b>{esc(orig[cid].parent.name)}</b> · {esc(VIEWS.get(m.get('view', ''), m.get('view', '')))}<br>
{esc(when(cid))}{' · ' + esc(source(cid)) if source(cid) else ''}<br><span class="fn">{esc(m.get('original_filename', ''))}</span>{f'<br><span class="note">{esc(note)}</span>' if note else ''}</figcaption></figure>"""


def same_view_photo(turtle, view, not_ids):
    """Another photo of `turtle` with the same view (another day if possible), for comparison."""
    d = {meta[c]["captured_date"] for c in not_ids if c in meta}
    pool = [p.stem for p in (TV / "data" / turtle).glob("*.jpg") if p.stem not in not_ids and p.stem in meta]
    same = [c for c in pool if meta[c]["view"] == view]
    other_day = [c for c in same if meta[c]["captured_date"] not in d]
    pick = (other_day or same or pool)
    return sorted(pick, key=lambda c: meta[c]["captured_date"])[-1] if pick else None


def item(uid, title, explain, photos, choices):
    btns = "".join(f'<button type="button" data-v="{esc(v)}">{esc(v)}</button>' for v in choices)
    return f"""<section class="item" data-id="{esc(uid)}"><h3>{title}</h3><p class="why">{explain}</p>
<div class="photos">{''.join(photos)}</div>
<div class="verdict"><span>Your verdict:</span>{btns}<input type="text" placeholder="notes (optional)" aria-label="notes"></div></section>"""


def dup_item(a, b, pixel_diff, same_turtle):
    na, nb = orig[a].parent.name, orig[b].parent.name
    sa, sb = source(a), source(b)
    if same_turtle:
        title = f"{esc(na)} · {esc(VIEWS.get(meta[a]['view'], meta[a]['view']))}: one picture, two dates ({esc(when(a)).split(' ·')[0]} and {esc(when(b)).split(' ·')[0]})"
        why = (f"These two files are the same picture (pixel difference {pixel_diff} once lined up; 0 = identical), "
               f"one {esc(sa)} and one {esc(sb)}, but their filenames give different dates, so one date is wrong.")
        labels = ("Copy 1's date is right", "Copy 2's date is right", "Different pictures", "Not sure")
    else:
        title = f"One picture filed under two turtles: {esc(na)} and {esc(nb)}"
        why = f"These two files are the same picture (pixel difference {pixel_diff}), filed under {esc(na)} and {esc(nb)}."
        labels = ("Copy 1 is right", "Copy 2 is right", "Different pictures", "Not sure")
    return item(f"duplicate-{a}-{b}", title, why,
                [card(a, "Copy 1", CONFIRMED.get(a, "")), card(b, "Copy 2", CONFIRMED.get(b, ""))], labels)


def main(out):
    flags = json.loads((OUT_DIR / "flags.json").read_text())
    sec = {k: [] for k in ("misfiled", "explained", "confirmed", "dup_turtles", "dup_dates", "no match")}
    seen_dups = set()
    for f in flags:
        p = f["photo"]
        m = meta.get(p, {})
        v = VIEWS.get(m.get("view", ""), m.get("view", ""))
        if f["kind"] == "misfiled":
            own_cmp = same_view_photo(f["turtle"], m.get("view", ""), {p} | set(FIXED))
            photos = [card(p, "Flagged photo", CONFIRMED.get(p, "")),
                      card(f["other_photo"], f"Strongest match ({f['other_best']} consistent)", CONFIRMED.get(f["other_photo"], "")),
                      *([card(own_cmp, f"A {f['turtle']} photo, same view")] if own_cmp else [])]
            explain = (f"This photo matches a <b>{esc(f['other_turtle'])}</b> photo with <b>{f['other_best']}</b> consistent matches, "
                       f"but its best match among {esc(f['turtle'])}'s other photos is only <b>{f['own_best']}</b>. "
                       f"Two different turtles never got above 6 in the reference set.")
            key = "misfiled"
            if p in FIXED:
                key = "confirmed"
            elif FIXED.get(f["other_photo"]) == f["turtle"]:
                key = "explained"
                explain += (f" <b>But</b> that {esc(f['other_turtle'])} photo is one you confirmed is really {esc(f['turtle'])}, "
                            f"so this flag goes away once that photo is moved.")
            sec[key].append(item(f"misfiled-{p}",
                f"{esc(f['turtle'])} · {esc(v)} · {esc(when(p))} — looks like <b>{esc(f['other_turtle'])}</b>",
                explain, photos, [f"It's {f['other_turtle']}", f"It's {f['turtle']} (correct)", "Not sure"]))
        elif f["kind"] == "duplicate":
            a, b = sorted((p, f["other"]), key=int)
            if (a, b) in seen_dups:
                continue
            seen_dups.add((a, b))
            same = orig[a].parent.name == orig[b].parent.name
            sec["dup_dates" if same else "dup_turtles"].append(dup_item(a, b, f["pixel_diff"], same))
        else:
            own_cmp = same_view_photo(f["turtle"], m.get("view", ""), {p})
            photos = [card(p, "Flagged photo"),
                      *([card(own_cmp, f"A {f['turtle']} photo, same view, another day")] if own_cmp else []),
                      card(f["other_photo"], f"Closest other turtle ({f['other_best']} consistent)")]
            sec["no match"].append(item(
                f"nomatch-{p}",
                f"{esc(f['turtle'])} · {esc(v)} · {esc(when(p))} — doesn't match {esc(f['turtle'])}'s other photos",
                f"Its best match among {esc(f['turtle'])}'s other photos is only <b>{f['own_best']}</b> consistent matches "
                f"(repeat photos of a turtle are usually 20 or more), and no other turtle matches it either (best {f['other_best']}). "
                f"Often a blurry, dark, distant or partial photo; sometimes the wrong turtle.",
                photos, [f"It's {f['turtle']} (poor photo)", "It's another turtle", "Not sure"]))

    n = {k: len(v) for k, v in sec.items()}
    blocks = []
    for key, head, intro in [
        ("misfiled", "New: likely filed under the wrong turtle", "Photos that match another turtle much more strongly than their own."),
        ("explained", "Flagged only because of a photo you already confirmed", "These will clear once the confirmed moves are made; shown so you can check."),
        ("no match", "Photos that don't match their own turtle", "Top and side photos with no real match to any other photo of their turtle, though it has other photos of that view. Usually poor photos; worth a glance."),
        ("dup_dates", "One picture stored twice with different dates", "Mostly a photo on the turtle's own record that is also in an encounter, with a different date in each filename. Same turtle, so identification isn't affected, but one date is wrong. These also made some of the accuracy tests look better than they were."),
        ("dup_turtles", "One picture filed under two turtles (you confirmed these)", ""),
        ("confirmed", "Already confirmed by you", "The photos you confirmed earlier, for completeness."),
    ]:
        if sec[key]:
            blocks.append(f'<h2>{head} <span class="count">{n[key]}</span></h2>' + (f'<p class="intro">{intro}</p>' if intro else "") + "".join(sec[key]))
    if not sec["misfiled"]:
        blocks.insert(0, '<h2>New: likely filed under the wrong turtle <span class="count">0</span></h2><p class="intro">None beyond the ones you already confirmed (listed at the bottom).</p>')
    page = PAGE.replace("{{BODY}}", "".join(blocks)).replace("{{SUMMARY}}", (
        f"No new misfiled photos · {n['no match']} photos with no match to their own turtle · "
        f"{n['dup_dates']} pictures stored twice under different dates · {n['dup_turtles'] + n['confirmed']} items you already confirmed"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB) {n}")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Photo Label Audit</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a64;--line:#e2e0d8;--accent:#2e6b3a;--pick:#2e6b3a;--pickink:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161714;--card:#20221e;--ink:#ecebe6;--muted:#a3a29a;--line:#34362f;--accent:#8fc79a;--pick:#8fc79a;--pickink:#10120f}}
:root[data-theme="dark"]{--bg:#161714;--card:#20221e;--ink:#ecebe6;--muted:#a3a29a;--line:#34362f;--accent:#8fc79a;--pick:#8fc79a;--pickink:#10120f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 60px}h1{margin:0 0 4px;font-size:24px}h2{margin:34px 0 4px;font-size:19px}
.count{font-weight:400;color:var(--muted)}.intro,.sub{color:var(--muted);margin:0 0 12px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:2}
.bar button{font:inherit;padding:7px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
#progress{color:var(--muted)}
.item{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.item h3{margin:0 0 4px;font-size:16px}.why{margin:0 0 10px;color:var(--muted)}
.photos{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.ph{margin:0}.ph img{width:100%;height:240px;object-fit:contain;background:#0001;border-radius:8px;display:block}
figcaption{font-size:13px;margin-top:5px}.role{display:block;color:var(--accent);font-weight:600}.fn{color:var(--muted);font-family:ui-monospace,monospace;font-size:12px}
.note{color:var(--accent);font-weight:600}
.verdict{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}.verdict span{color:var(--muted)}
.verdict button{font:inherit;padding:6px 11px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
.verdict button.on{background:var(--pick);color:var(--pickink);border-color:var(--pick)}
.verdict input{flex:1;min-width:180px;font:inherit;padding:6px 9px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--ink)}
.item.done{border-color:var(--accent)}
</style></head><body><main>
<h1>Photo label audit</h1>
<p class="sub">Reference photos checked against each other by shell pattern (SIFT). {{SUMMARY}}. Times come from the original photo filenames; search Airtable for the filename to find a photo. Your answers stay in this browser until you download them.</p>
<div class="bar"><span id="progress"></span><button type="button" id="dl">Download my answers (CSV)</button></div>
{{BODY}}
</main><script>
const KEY = 'turtle-audit-v1';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const store = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} };
const items = [...document.querySelectorAll('.item')];
function progress(){ document.getElementById('progress').textContent = `${items.filter(i => saved[i.dataset.id]?.v).length} of ${items.length} answered`; }
for (const it of items) {
  const s = saved[it.dataset.id] || {};
  const input = it.querySelector('input');
  input.value = s.note || '';
  for (const b of it.querySelectorAll('.verdict button')) {
    if (b.dataset.v === s.v) { b.classList.add('on'); it.classList.add('done'); }
    b.onclick = () => {
      it.querySelectorAll('.verdict button').forEach(x => x.classList.toggle('on', x === b));
      it.classList.add('done');
      saved[it.dataset.id] = {...(saved[it.dataset.id] || {}), v: b.dataset.v}; store(); progress();
    };
  }
  input.oninput = () => { saved[it.dataset.id] = {...(saved[it.dataset.id] || {}), note: input.value}; store(); };
}
progress();
document.getElementById('dl').onclick = () => {
  const q = s => '"' + String(s ?? '').replace(/"/g, '""') + '"';
  const rows = [['item', 'title', 'filenames', 'verdict', 'notes']];
  for (const it of items) {
    const fns = [...it.querySelectorAll('.fn')].map(e => e.textContent).join(' | ');
    rows.push([it.dataset.id, it.querySelector('h3').textContent, fns, saved[it.dataset.id]?.v, saved[it.dataset.id]?.note]);
  }
  const blob = new Blob([rows.map(r => r.map(q).join(',')).join('\\n')], {type: 'text/csv'});
  const a = Object.assign(document.createElement('a'), {href: URL.createObjectURL(blob), download: 'turtle_audit_answers.csv'});
  a.click();
};
</script></body></html>"""

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop/turtle-review/audit.html")
