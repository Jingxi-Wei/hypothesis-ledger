"""Combine the 3 sample types into ONE mixed LLaMA-Factory training file (no separate training /
curriculum — just well-marked targets, trained together) + write dataset_info.json.

Run after export.py. Upload dataset/sft/ to a rented GPU and train with the config in
src/configs/qwen_qlora_sft.yaml.
"""
import json
import random
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _schema  # noqa: E402  — field-schema skins (surface-invariance augmentation)

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "dataset" / "samples"
SFT = ROOT / "dataset" / "sft"
app = typer.Typer(add_completion=False)


@app.command()
def main(seed: int = typer.Option(42, "--seed"),
         max_tokens: int = typer.Option(8192, "--max-tokens",
                                        help="drop samples whose est. tokens exceed this (0 = keep all). "
                                             "Set == training cutoff_len so NOTHING truncates: a truncated "
                                             "target teaches the model to write half a sentence."),
         chars_per_token: float = typer.Option(3.5, "--chars-per-token", help="conservative est (code-heavy)"),
         keep_raw_leak: bool = typer.Option(False, "--keep-raw-leak",
                                            help="include OLD raw-leak-protocol correction samples (default: drop — "
                                                 "they'll be re-collected clean under the sanitized protocol)")) -> None:
    SFT.mkdir(parents=True, exist_ok=True)
    rows, by = [], {}
    dropped = {}       # by type — samples too long to fit the training window without truncation
    dropped_leak = {}  # by source — OLD raw-leak-protocol correction samples (dropped unless --keep-raw-leak)
    skinned = 0        # rows rewritten into an alternate field schema (surface-invariance augmentation)
    # merge every run's samples: dataset/samples/<run_id>/{audit,propose,fix,probe}.jsonl (r1 + pro1 + ...)
    for t in ("audit", "propose", "fix", "probe"):
        for p in sorted(SAMP.glob(f"*/{t}.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                s = json.loads(line)
                if not keep_raw_leak and s.get("protocol") == "raw_leak":
                    src = s.get("source") or "?"  # correction sample from an old raw-leak trajectory — re-collect, don't train
                    dropped_leak[src] = dropped_leak.get(src, 0) + 1
                    continue
                m = s["messages"]
                inp, tgt = m[0]["content"], m[1]["content"]
                if s["type"] != "fix":  # fix target is a raw patch — no labeled sections to rename
                    skin = _schema.pick(f'{s.get("instance_id", "")}|{s.get("hypothesis_id", "")}|{s["type"]}')
                    if skin:  # rename the target's section labels + tell the input which labels to use
                        nt = _schema.note(_schema.labels_in_target(tgt), skin)
                        if nt:
                            inp, tgt = inp + nt, _schema.apply_skin(tgt, skin)
                            skinned += 1
                if max_tokens > 0 and (len(inp) + len(tgt)) / chars_per_token > max_tokens:
                    dropped[t] = dropped.get(t, 0) + 1  # would truncate -> drop (mostly monster gold-patch fixes)
                    continue
                rows.append({"conversations": [{"from": "human", "value": inp},
                                               {"from": "gpt", "value": tgt}],
                             "type": s["type"], "turning_point": s.get("turning_point", False),
                             "source": s.get("source"),          # explore/self_rescue/oracle
                             "protocol": s.get("protocol", "clean")})
                by[t] = by.get(t, 0) + 1
    # action accounting — the balance dashboard: hold(future) / refine(self_rescue) / redirect(oracle) / probe / solve.
    # correction_frac drives the Pro-vs-Verified stop rule; watch it against the eval's premature_refute / missed_flaw.
    def _action(r: dict) -> str:
        if r["type"] == "probe":
            return "probe"
        if r["type"] == "fix":
            return "fix"
        if r.get("source") == "oracle":
            return "redirect"
        if r.get("source") == "self_rescue":
            return "refine"
        return "solve"  # explore-sourced audit/propose, incl. every self_solved trajectory
    act: dict[str, int] = {}
    for r in rows:
        act[_action(r)] = act.get(_action(r), 0) + 1
    n = len(rows) or 1
    random.Random(seed).shuffle(rows)
    (SFT / "train_sharegpt.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (SFT / "dataset_info.json").write_text(json.dumps({
        "hl_sft": {
            "file_name": "train_sharegpt.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        }
    }, indent=2), encoding="utf-8")
    tp = sum(1 for r in rows if r["turning_point"])
    correction_frac = round((act.get("redirect", 0) + act.get("refine", 0)) / n, 3)
    (SFT / "accounting.json").write_text(json.dumps(
        {"total": len(rows), "by_type": by, "by_action": act,
         "action_pct": {k: round(100 * v / n) for k, v in sorted(act.items())},
         "correction_frac": correction_frac, "schema_skinned": skinned,
         "dropped_over_len": dropped, "dropped_raw_leak": dropped_leak},
        indent=2), encoding="utf-8")
    dz = f" | DROPPED {sum(dropped.values())} over-{max_tokens}tok {dropped}" if dropped else ""
    lz = f" | DROPPED {sum(dropped_leak.values())} raw_leak {dropped_leak}" if dropped_leak else ""
    act_pct = {k: f"{v}({round(100 * v / n)}%)" for k, v in sorted(act.items())}
    print(f"[prep_sft] {len(rows)} rows by type={by} | actions={act_pct} correction_frac={correction_frac} | "
          f"schema-skinned={skinned} ({round(100*skinned/n)}%) | turning_point audits={tp}{dz}{lz} -> {SFT / 'train_sharegpt.jsonl'}")


if __name__ == "__main__":
    app()
