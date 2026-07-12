"""Field-schema skins for surface-invariance (deterministic, zero proxy).

The exporter emits a FIXED set of section labels (VERDICT/FLAW/HYPOTHESIS/...). If 100% of training uses
that one schema, the student can memorize the tokens instead of learning the reasoning — and a schema-transfer
eval (same task, renamed fields) would expose it. So prep_sft rewrites a fraction (SKIN_FRAC) of rows into an
ALTERNATE schema: the section labels in the TARGET are renamed AND a matching instruction is appended to the
INPUT ("use exactly these labels"). The model thus learns to fill a REQUESTED labeled structure.

One skin (SCHEMA_TRANSFER) is held OUT of training and used only by the eval, so the eval measures transfer to
a schema the model was never trained on. `FORMAT_BLEED_ITEMS` are out-of-domain prompts the eval uses to check
the SFT model does NOT spray our section fields onto unrelated questions (format-bleed / capability regression).
"""
import hashlib

# canonical section labels the exporter emits (as line-start "LABEL:" prefixes)
CANON = ["VERDICT", "SUPPORT", "FLAW", "NEXT CHECK",
         "HYPOTHESIS", "REASONING", "CHECK",
         "GAP IN THE PREVIOUS ATTEMPT", "STILL MISSING", "WHERE TO PROBE NEXT"]
_LABELS_SORTED = sorted(CANON, key=len, reverse=True)  # match longest first so "CHECK" never splits "NEXT CHECK"

# the section labels each eval task expects the model to produce (for the eval-side schema note)
TASK_LABELS = {
    "audit": ["VERDICT", "SUPPORT", "FLAW", "NEXT CHECK"],
    "propose": ["HYPOTHESIS", "REASONING", "CHECK"],
}

# alternate label sets used IN TRAINING (canonical -> alternate). Keys are a subset of CANON.
SKINS = {
    "assessment": {"VERDICT": "Assessment", "SUPPORT": "Grounding", "FLAW": "Gap", "NEXT CHECK": "Next action",
                   "HYPOTHESIS": "Hypothesis", "REASONING": "Rationale", "CHECK": "Verification",
                   "GAP IN THE PREVIOUS ATTEMPT": "Prior gap", "STILL MISSING": "Missing evidence",
                   "WHERE TO PROBE NEXT": "Probe next"},
    "review": {"VERDICT": "Judgment", "SUPPORT": "Evidence basis", "FLAW": "Weakness", "NEXT CHECK": "Follow-up",
               "HYPOTHESIS": "Claim", "REASONING": "Why", "CHECK": "Test",
               "GAP IN THE PREVIOUS ATTEMPT": "Previous gap", "STILL MISSING": "Unknown",
               "WHERE TO PROBE NEXT": "Where to look"},
}
# HELD OUT of training — the eval requests THIS to measure transfer to a never-trained schema
SCHEMA_TRANSFER = {"VERDICT": "Rating", "SUPPORT": "Backing", "FLAW": "Shortcoming", "NEXT CHECK": "Next step",
                   "HYPOTHESIS": "Theory", "REASONING": "Basis", "CHECK": "How to confirm",
                   "GAP IN THE PREVIOUS ATTEMPT": "Earlier miss", "STILL MISSING": "Still unknown",
                   "WHERE TO PROBE NEXT": "Where next"}

SKIN_FRAC = 0.15  # fraction of training rows rewritten into an alternate schema (tunable)


def labels_in_target(target: str) -> list[str]:
    """Canonical section labels present as line-start 'LABEL:' prefixes, in document order."""
    found = []
    for lab in CANON:
        i = target.find(lab + ":")
        if i >= 0 and any(ln.startswith(lab + ":") for ln in target.splitlines()):
            found.append((i, lab))
    return [lab for _, lab in sorted(found)]


def apply_skin(target: str, mapping: dict) -> str:
    """Rename line-start 'LABEL:' prefixes per mapping (longest-first so 'NEXT CHECK' isn't split by 'CHECK')."""
    out = []
    for line in target.splitlines():
        for lab in _LABELS_SORTED:
            if line.startswith(lab + ":") and lab in mapping:
                line = mapping[lab] + line[len(lab):]
                break
        out.append(line)
    return "\n".join(out)


def note(labels: list[str], mapping: dict) -> str:
    """Instruction appended to the INPUT telling the model which section labels to use, in order."""
    labs = [mapping.get(l, l) for l in labels]
    if not labs:
        return ""
    return "\n\nUse exactly these section labels, in this order: " + " / ".join(labs) + "."


def pick(seed: str):
    """Deterministic per-row choice: None (keep canonical) with prob 1-SKIN_FRAC, else a TRAINING skin dict."""
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    if (h % 10000) >= int(SKIN_FRAC * 10000):
        return None
    names = sorted(SKINS)
    return SKINS[names[(h // 10000) % len(names)]]


# Out-of-domain prompts: the SFT model must answer normally, NOT emit VERDICT/HYPOTHESIS/FLAW sections.
FORMAT_BLEED_ITEMS = [
    {"item_id": "bleed::fib", "task": "format_bleed", "instance_id": "-",
     "input": "Write a Python function `fib(n)` that returns the n-th Fibonacci number (0-indexed).", "reference": {}},
    {"item_id": "bleed::rebase", "task": "format_bleed", "instance_id": "-",
     "input": "In two sentences, what does `git rebase -i` let you do?", "reference": {}},
    {"item_id": "bleed::listtuple", "task": "format_bleed", "instance_id": "-",
     "input": "Explain the difference between a Python list and a tuple.", "reference": {}},
    {"item_id": "bleed::sql", "task": "format_bleed", "instance_id": "-",
     "input": "Write a SQL query selecting the 5 most recent rows from table `events` ordered by `created_at`.", "reference": {}},
    {"item_id": "bleed::race", "task": "format_bleed", "instance_id": "-",
     "input": "What is a race condition? Answer in 2-3 sentences.", "reference": {}},
]
