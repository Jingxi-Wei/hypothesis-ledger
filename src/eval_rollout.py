"""Agentic rollout eval: let the model REDO problems end-to-end and see whether it self-corrects.

No oracle, no directions, ever. Two feedback arms (run both; 2026-07-07 design):
  --feedback none     the model is on its own: verify with its OWN repro runs, submit once, episode ends.
                      Primary read: premature-submit / self-testing behavior.
  --feedback binary   a failed submit gets ONE BIT back ("did not resolve", no details), up to --max-retries.
                      Primary read: redirect-after-failure vs thrash (unchanged resubmits).

Tiers (build with select_rollout_sets.py): rollout_seen.json = instances the SFT data came from
(absorption check — their fix samples contained the gold patch, so resolve here is NOT a capability claim;
behavior metrics are the read). rollout_unseen.json = pro_test holdout (the generalization tier).

Model = any OpenAI-compatible endpoint. For the real thing: vLLM on the GPU box serving Qwen base / the
SFT adapter, then --model openai/<served-name> --api-base http://<gpu>:8000/v1. Same agent scaffold
(swebench_hypo prompt) for BOTH arms and BOTH models — we compare behavior, not prompting.

Trajectories land in dataset/raw/<iid>/<run_id>/ exactly like collection, so compress + the behavior
grader (eval_rollout_grade.py) work unchanged. Resumable: instances with outcome.json are skipped.

  python src/eval_rollout.py --instances-file dataset/splits/rollout_seen.json --run-id rollB_seen_sft \
      --feedback binary --model openai/qwen-sft --api-base http://127.0.0.1:8000/v1
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402  — reuses the collection harness (env setup, eval, docker hygiene)
from collect import CollectorAgent, eval_in_container  # noqa: E402
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_config_from_spec, get_sb_environment  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from datasets import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
app = typer.Typer(add_completion=False)

_BINARY_FB = ("Rollout feedback (binary): your patch did not resolve the issue. No further details are "
              "available. Reconsider your HYPOTHESIS against the evidence you have gathered and continue.")


class RolloutAgent(CollectorAgent):
    """CollectorAgent with the crutches removed: eval feedback is one bit at most."""

    def __init__(self, *a, feedback: str = "binary", max_retries: int = 3, **kw):
        super().__init__(*a, **kw)
        self.feedback_mode = feedback
        self.max_retries = max_retries
        self.retries = 0
        self.unchanged_resubmits = 0

    def _stage_label(self) -> str:  # LimitsExceeded path in collect(): solved without a final submit
        return "resolved_nosubmit"

    def _on_submit(self, patch: str) -> dict:
        prev = self.last_patch
        self.last_patch = patch or self.last_patch
        unchanged = patch == prev or not (patch or "").strip()
        ev = eval_in_container(self.env, self.instance)
        self.evals.append({"stage": f"attempt_{self.retries + 1}", "resolved": ev["resolved"],
                           "f2p_pass": ev["f2p_pass"], "p2p_pass": ev["p2p_pass"]})
        if ev["resolved"]:
            self.outcome = "resolved_first" if self.retries == 0 else "resolved_retry"
            return {"output": "Submission recorded. Episode complete.", "returncode": 0, "exception_info": ""}
        if self.feedback_mode == "none":
            self.outcome = "failed_first"  # one shot: the verdict is never shown to the model
            return {"output": "Submission recorded. Episode complete.", "returncode": 1, "exception_info": ""}
        if unchanged:
            self.unchanged_resubmits += 1
        if self.retries < self.max_retries:
            self.retries += 1
            return {"output": _BINARY_FB + f" (attempt {self.retries} of {self.max_retries} retries used)",
                    "returncode": 1, "exception_info": ""}
        self.outcome = "failed_retries"
        return {"output": "Out of attempts. Episode complete.", "returncode": 1, "exception_info": ""}


def rollout_one(instance: dict, run_id: str, feedback: str, max_retries: int,
                model_name: str, api_base: str, api_key: str) -> dict:
    config = get_config_from_spec(collect.CONFIG)
    if collect._is_pro():
        import pro
        instance = pro.normalize_pro(instance)
        ec = config.setdefault("environment", {})
        ec["cwd"], ec["timeout"], ec["run_args"] = "/app", 1800, ["--rm", "--entrypoint", ""]
        ec["pull_timeout"] = 7200
        ag = config.setdefault("agent", {})
        for k in ("system_template", "instance_template"):
            if isinstance(ag.get(k), str):
                ag[k] = ag[k].replace("/testbed", "/app")
    out_dir = ROOT / "dataset" / "raw" / instance["instance_id"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_cfg = {k: v for k, v in config.get("agent", {}).items() if k != "agent_class"}
    agent_cfg["output_path"] = str(out_dir / "trajectory.json")
    mc = config.get("model", {})
    mc["model_name"] = model_name
    if api_base:
        mc["api_base"] = api_base
    if api_key:
        mc["api_key"] = api_key
    extra_body = mc.setdefault("model_kwargs", {}).setdefault("extra_body", {})
    extra_body.pop("reasoning_effort", None)
    extra_body.pop("speed", None)
    model = get_model(config=mc)
    env = get_sb_environment(config, instance)
    try:
        agent = RolloutAgent(model, env, instance=instance, feedback=feedback, max_retries=max_retries, **agent_cfg)
        outcome = agent.collect(instance["problem_statement"])
        summary = {"instance_id": instance["instance_id"], "outcome": outcome, "evals": agent.evals,
                   "feedback_mode": feedback, "max_retries": max_retries, "retries_used": agent.retries,
                   "unchanged_resubmits": agent.unchanged_resubmits, "model": model_name,
                   "n_messages": len(agent.messages)}
        (out_dir / "outcome.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[rollout] {instance['instance_id']} -> {outcome} "
              f"(attempts {len(agent.evals)}, unchanged {agent.unchanged_resubmits})", flush=True)
        return summary
    finally:
        try:  # same Windows-safe, per-container cleanup as collect_one
            cid = getattr(env, "container_id", None)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
            else:
                env.cleanup()
        except Exception:
            pass


@app.command()
def main(instances_file: str = typer.Option(..., "--instances-file"),
         run_id: str = typer.Option(..., "--run-id", help="e.g. rollB_seen_sft / rollA_unseen_base"),
         dataset: str = typer.Option("pro", "--dataset", help="pro | verified (rollout sets are Pro by default)"),
         feedback: str = typer.Option("binary", "--feedback", help="none | binary"),
         max_retries: int = typer.Option(3, "--max-retries"),
         model: str = typer.Option(os.environ.get("EVAL_MODEL", ""), "--model",
                                   help="litellm name, e.g. openai/qwen-sft (vLLM served name)"),
         api_base: str = typer.Option(os.environ.get("EVAL_API_BASE", ""), "--api-base"),
         api_key: str = typer.Option(os.environ.get("EVAL_API_KEY", "pwd"), "--api-key"),
         limit: int = typer.Option(0, "--limit")) -> None:
    if feedback not in ("none", "binary"):
        raise typer.BadParameter("--feedback must be none|binary")
    if not model:
        raise typer.BadParameter("--model (or EVAL_MODEL) is required — point it at the vLLM endpoint's served name")
    os.environ["DATASET"] = dataset  # collect._is_pro() reads this at runtime (same convention as run_batch)
    name = {"pro": "ScaleAI/SWE-bench_Pro", "verified": "princeton-nlp/SWE-bench_Verified"}[dataset]
    ds = {i["instance_id"]: i for i in load_dataset(DATASET_MAPPING.get(name, name), split="test")}
    ids = json.loads(Path(instances_file).read_text(encoding="utf-8"))
    if limit:
        ids = ids[:limit]
    done = err = 0
    for iid in ids:
        if iid not in ds:
            print(f"[skip] {iid} not in dataset")
            continue
        if (ROOT / "dataset" / "raw" / iid / run_id / "outcome.json").exists():
            done += 1
            continue
        try:
            rollout_one(ds[iid], run_id, feedback, max_retries, model, api_base, api_key)
            done += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err += 1
            print(f"[rollout] {iid} ERROR {type(e).__name__}: {str(e)[:160]} (resumable — rerun to retry)", flush=True)
    print(f"[rollout done] run={run_id} ok/skipped={done} errors={err} | next: python src/compress.py --run-id {run_id}"
          f" ; then python src/eval_rollout_grade.py --runs <base_run>,{run_id}")


if __name__ == "__main__":
    app()
