"""Quick trajectory inspector: HYPOTHESIS lines, step count, submission."""
import json
import sys
from pathlib import Path


def text_of(content) -> str:
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    return content or ""


def main(path: str) -> None:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    msgs = d["messages"]
    info = d.get("info", {})
    asst = [m for m in msgs if m.get("role") == "assistant"]
    print("exit_status   :", info.get("exit_status"))
    print("submission len:", len(info.get("submission", "") or ""))
    print("api_calls     :", info.get("model_stats", {}).get("api_calls"))
    print("assistant msgs:", len(asst), "(= steps/phases candidate)")
    hyps = []
    for m in msgs:
        for line in text_of(m.get("content", "")).splitlines():
            if "HYPOTHESIS:" in line:
                hyps.append(line.strip())
    print(f"HYPOTHESIS lines: {len(hyps)}")
    for h in hyps:
        print("   -", h[:200])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset/raw/_smoke/astropy-12907.traj.json")
