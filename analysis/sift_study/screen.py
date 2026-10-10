"""Screening table: every variant on the fixed 100-photo sample (50 reference queries,
30 known + 20 new field photos). Run lab_sift.py for each config first.

    python screen.py base,mutn@0.8,mutg@0.8,lnbnn_ref
"""
import sys, json
from lab_eval import load, table, combine
cfgs = [{}, {"root": True}, {"root": True, "mask": "ellipse"}, {"root": True, "pre": "lab_b"}, {"pre": "lab_b"}, {"mask": "ellipse"}, {"root": True, "mask": "ellipse", "pre": "lab_b"}]
S = {json.dumps(c): load(c) for c in cfgs}
kinds = sys.argv[1].split(",")
ent = [(k2, S[k2], k, "max") for k in kinds for k2 in S]
for a, b in [("{}", '{"pre": "lab_b"}'), ('{"root": true}', '{"root": true, "pre": "lab_b"}')]:
    for k in ("mutn@0.8", "mutg@0.8"):
        C, tag = combine(S[a], S[b], k)
        ent.append((f"{a}+{b}"[:34], C, tag, "max"))
table(ent)
