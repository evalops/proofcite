from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

try:
    import dspy  # type: ignore
    _HAS_DSPY = True
except Exception:
    dspy = None  # type: ignore
    _HAS_DSPY = False

from .dspy_judge import _ensure, Judge


def load_labeled(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser(description="Optimize the DSPy Judge with few-shot demos from labeled data.")
    ap.add_argument("--train", required=True, help="JSONL labeled data: {requirement, evidence_lines, verdict, reason}")
    ap.add_argument("--out", default="proofcite/evals/judge_demos.jsonl")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if not _HAS_DSPY:
        print("DSPy not installed. pip install dspy-ai")
        return

    _ensure(model=args.model)
    # Prepare trainset
    trainset = []
    for ex in load_labeled(args.train):
        req = ex["requirement"]
        lines = ex.get("evidence_lines", [])
        verdict = ex.get("verdict", "Unverifiable")
        reason = ex.get("reason", "")
        result = json.dumps({"verdict": verdict, "reason": reason})
        trainset.append({
            "requirement": req,
            "evidence": "\n".join(lines),
            "result_json": result,
        })

    program = dspy.Predict(Judge)
    optimizer = dspy.BootstrapFewShot(metric=lambda pred, gold: 1.0 if getattr(pred, 'result_json', '') else 0.0)  # type: ignore
    compiled = optimizer.compile(program, trainset=trainset)

    # Persist demos if available
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    demos = getattr(compiled, "demos", None)
    if demos:
        with out.open("w", encoding="utf-8") as f:
            for d in demos:
                f.write(json.dumps(d) + "\n")
        print(f"Saved judge demos to {out}")
    else:
        print("No demos produced; the judge will still run without demos.")


if __name__ == "__main__":
    main()

