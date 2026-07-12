"""Build a HARD-problem pool from SWE-bench Full for scaled data generation.

SWE-bench Full has NO human difficulty labels (unlike Verified), so 'hard' is a PROXY = gold-patch complexity
(source files touched + lines changed + # of FAIL_TO_PASS). Imperfect (a small patch can be a hard-to-find bug),
but biases the pool toward non-trivial fixes -> better oracle_redirected (gold) yield than random.

Excludes SWE-bench Verified (already collected) and Lite (the easy curated subset). Keeps only eval-MAP-covered
instances. Saves dataset/splits/full_hard.json = instance_ids RANKED hardest-first (run_batch --limit N takes the top N).

  python src/select_full.py               # build the pool
  python src/select_full.py --min-score 25   # tighter hard threshold
"""
import json
import re
import sys
from pathlib import Path

import typer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_wincompat"))
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
app = typer.Typer(add_completion=False)


def _jl(x):
    return json.loads(x) if isinstance(x, str) else x


def complexity(inst: dict) -> tuple[int, int, int, int]:
    p = inst["patch"]
    files = set(re.findall(r"^\+\+\+ b/(.+)", p, re.M))
    src = [f for f in files if "/test" not in f and not f.split("/")[-1].startswith("test")]
    lines = len([l for l in p.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))])
    f2p = len(_jl(inst["FAIL_TO_PASS"]))
    # each component CAPPED so none dominates: multi-file fix (up to 5) + patch size (up to 100 lines) +
    # a small f2p tiebreak. Huge-f2p instances (broad-blast-radius, slow eval) are NOT treated as 'hard'.
    score = min(len(src), 5) * 15 + min(lines, 100) + min(f2p, 4) * 3
    return score, len(src), lines, f2p


@app.command()
def main(min_score: int = typer.Option(0, "--min-score", help="optional hardness-proxy cutoff; 0 = keep all, ranked"),
         max_f2p: int = typer.Option(60, "--max-f2p", help="drop instances with >N FAIL_TO_PASS (slow eval / broad blast radius)")) -> None:
    full = list(load_dataset("princeton-nlp/SWE-bench", split="test"))
    ver = {i["instance_id"] for i in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    lite = {i["instance_id"] for i in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}

    def in_map(i):
        try:
            return i["version"] in M[i["repo"]]
        except Exception:
            return False

    cand = [i for i in full if i["instance_id"] not in ver and i["instance_id"] not in lite and in_map(i)]
    scored = sorted(((complexity(i), i) for i in cand), key=lambda x: -x[0][0])
    hard = [(s, i) for (s, i) in scored if s[0] >= min_score and s[3] <= max_f2p]
    ids = [i["instance_id"] for (s, i) in hard]

    out = ROOT / "dataset" / "splits" / "full_hard.json"
    out.write_text(json.dumps(ids, indent=0), encoding="utf-8")
    import collections
    repos = collections.Counter(i["repo"] for (s, i) in hard)
    print(f"candidates (Full - Verified - Lite, eval-covered): {len(cand)}")
    print(f"HARD (score >= {min_score}): {len(hard)} -> {out}")
    print("by repo:", dict(repos.most_common(12)))
    print("hardest 5:", [(i["instance_id"], s[1], s[2], s[3]) for (s, i) in hard[:5]], "(id, src_files, lines, f2p)")


if __name__ == "__main__":
    app()
