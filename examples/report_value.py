import argparse
import json
import os
import time
from pathlib import Path

from proofcite.regulatory import run_regulatory_eval


def main():
    ap = argparse.ArgumentParser(description="Generate a proof-of-value report: general evals + regulatory proofs.")
    ap.add_argument("--docs_reg", default="proofcite/examples/regulatory/*.txt")
    ap.add_argument("--rules", default="proofcite/examples/regulatory/rules_extended.jsonl")
    ap.add_argument("--outdir", default="proofcite/out")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Regulatory run (baseline)
    reg_bundle = run_regulatory_eval(
        mode="baseline",
        docs_glob=args.docs_reg,
        rules_path=args.rules,
        default_threshold=0.35,
        rerank="hybrid",
        span_max_gap=1,
    )

    # Persist artifacts
    report = {"regulatory": reg_bundle["summary"]}
    report_path = out_dir / f"value_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    proofs_dir = out_dir / f"proofs_{ts}"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(reg_bundle["results"], start=1):
        (proofs_dir / f"rule_{i:02d}_{r['rule'].get('id','rule')}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Regulatory summary:", json.dumps(reg_bundle["summary"], indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved proofs: {proofs_dir}")


if __name__ == "__main__":
    main()
