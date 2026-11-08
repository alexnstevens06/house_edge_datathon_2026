"""Copy tune/confirm ids into this study. Does not score."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
STUDY = os.path.abspath(os.path.join(HERE, ".."))
src = os.path.join(REPO, "runs", "splits.json")
digest = hashlib.sha256(open(src, "rb").read()).hexdigest()
raw = json.load(open(src))
out = {
    "source": "runs/splits.json",
    "source_sha256": digest,
    "tune": raw["tune"],
    "confirm_ids_not_scored": raw["confirm"],
    "note": "Confirm ids are stored so the split file is complete. This study does not score them.",
}
path = os.path.join(STUDY, "SPLIT_IDS.json")
json.dump(out, open(path, "w"), indent=1)
print(path, digest, "tune", len(out["tune"]))
