"""Checks every evidence string in testset_7_1.json occurs in data/papers_txt (the text the retriever indexes)."""
import glob, json, re
from pathlib import Path
here = Path(__file__).parent
norm = lambda s: re.sub(r"\s+", " ", s.replace("-\n", "")).lower()
texts = {Path(f).name: norm(Path(f).read_text(errors="replace")) for f in glob.glob(str(here.parent / "data/papers_txt/*.txt"))}
bad = 0
for t in json.loads((here / "testset_7_1.json").read_text()):
    for e in t["evidence"]:
        hits = [f for f, x in texts.items() if norm(e) in x]
        if not hits:
            bad += 1; print("MISSING", t["id"], repr(e))
print("evidence strings all found" if not bad else f"{bad} missing")
