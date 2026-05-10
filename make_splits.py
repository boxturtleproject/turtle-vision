"""Build train/valid/test split of carapace photos for individual-turtle ID.

Filters:
  - category == 'carapace'
  - media_type == 'photo'   (illustrations excluded)
  - per turtle: keep only individuals with >= 4 qualifying images

Per-turtle split (seeded, deterministic):
  n_valid = max(1, round(n * 0.15))
  n_test  = max(1, round(n * 0.15))
  n_train = n - n_valid - n_test         # >= 2 because we required n >= 4
"""
import csv, random
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SRC = DATA / "classifications.csv"
OUT = DATA / "splits.csv"
SEED = 42
MIN_TOTAL = 4
MIN_TRAIN = 2

def split_counts(n):
    n_valid = max(1, round(n * 0.15))
    n_test  = max(1, round(n * 0.15))
    n_train = n - n_valid - n_test
    assert n_train >= MIN_TRAIN, (n, n_train, n_valid, n_test)
    return n_train, n_valid, n_test

def main():
    rows = list(csv.DictReader(SRC.open()))
    total = len(rows)

    # Apply filters
    cara = [r for r in rows if r["category"] == "carapace" and r["media_type"] == "photo"]
    illustrations = [r for r in rows if r["media_type"] == "illustration"]
    other_cats = [r for r in rows if r["category"] != "carapace" and r["media_type"] == "photo"]

    by_turtle = defaultdict(list)
    for r in cara:
        by_turtle[r["turtle_name"]].append(r)

    # All turtles seen in the source (any category)
    all_turtles = sorted({r["turtle_name"] for r in rows})

    kept_turtles, filtered_turtles = [], []
    for name in all_turtles:
        n = len(by_turtle.get(name, []))
        if n >= MIN_TOTAL:
            kept_turtles.append((name, n))
        else:
            filtered_turtles.append((name, n))

    rng = random.Random(SEED)
    out_rows = []
    per_turtle_split = []
    for name, _ in kept_turtles:
        items = sorted(by_turtle[name], key=lambda r: int(r["capture_id"]))
        rng.shuffle(items)
        n = len(items)
        n_train, n_valid, n_test = split_counts(n)
        train = items[:n_train]
        valid = items[n_train:n_train + n_valid]
        test  = items[n_train + n_valid:]
        for r in train: out_rows.append({**r, "split": "train"})
        for r in valid: out_rows.append({**r, "split": "valid"})
        for r in test:  out_rows.append({**r, "split": "test"})
        per_turtle_split.append((name, n, n_train, n_valid, n_test))

    # Write output CSV (preserve original cols + 'split')
    fields = ["capture_id", "turtle_name", "file_path", "category", "media_type", "confidence", "split"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # ---- Report ----
    def n_unique(rows): return len({r["turtle_name"] for r in rows})

    print(f"Source rows: {total}")
    print(f"  carapace photos:           {len(cara):>4}  ({n_unique(cara)} individuals)")
    print(f"  illustrations (excluded):  {len(illustrations):>4}  ({n_unique(illustrations)} individuals)")
    print(f"  other photo categories:    {len(other_cats):>4}  "
          f"(habitat/plastron/other_closeup; {n_unique(other_cats)} individuals)")
    print()
    print(f"Individuals total in source: {len(all_turtles)}")
    print(f"  kept (>= {MIN_TOTAL} carapace photos):     {len(kept_turtles)}")
    print(f"  filtered (< {MIN_TOTAL} carapace photos):  {len(filtered_turtles)}")
    print()
    print("Filtered-out individuals (carapace-photo count):")
    by_count = defaultdict(list)
    for name, n in filtered_turtles: by_count[n].append(name)
    for n in sorted(by_count):
        print(f"  n={n}: {len(by_count[n]):>2}  -> {', '.join(sorted(by_count[n]))}")
    print()
    splits = defaultdict(int)
    for r in out_rows: splits[r["split"]] += 1
    print(f"Output: {OUT.relative_to(Path('/Users/boaz/Code/boxturtle'))}")
    print(f"  total rows: {sum(splits.values())}  "
          f"(train={splits['train']}, valid={splits['valid']}, test={splits['test']})")
    print(f"  individuals (classes): {len(kept_turtles)}")
    print()
    print("Per-individual split (sorted by total desc):")
    print(f"  {'name':<22} {'total':>5} {'train':>5} {'valid':>5} {'test':>5}")
    for name, n, ntr, nv, nte in sorted(per_turtle_split, key=lambda x: -x[1]):
        print(f"  {name:<22} {n:>5} {ntr:>5} {nv:>5} {nte:>5}")

if __name__ == "__main__":
    main()
