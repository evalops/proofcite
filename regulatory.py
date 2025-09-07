from __future__ import annotations
import argparse
import glob
import json
import re
from typing import Any, Dict, Optional

from .core import ProofCite

try:
    from .dspy_variant import ProofCiteDSPy  # type: ignore
    _HAS_DSPY = True
except Exception:
    ProofCiteDSPy = None  # type: ignore
    _HAS_DSPY = False

try:
    from .evals.dspy_judge import DSPyJudge  # type: ignore
    _HAS_DSPY_JUDGE = True
except Exception:
    DSPyJudge = None  # type: ignore
    _HAS_DSPY_JUDGE = False


def load_rules(path: str):
    rules = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rules.append(json.loads(line))
    return rules


def run_regulatory_eval(
    mode: str,
    docs_glob: str,
    rules_path: str,
    default_threshold: float = 0.35,
    rerank: str = "none",
    span_max_gap: int = 1,
    judge: Optional[str] = None,
) -> Dict[str, Any]:
    paths = sorted(glob.glob(docs_glob))
    if mode == "baseline":
        pc = ProofCite()
    elif mode == "dspy":
        if not _HAS_DSPY:
            raise RuntimeError("DSPy not installed; `pip install dspy-ai`.")
        pc = ProofCiteDSPy()
    else:
        raise ValueError("mode must be 'baseline' or 'dspy'")

    pc.add_documents(paths)
    pc.build()

    rules = load_rules(rules_path)
    results = []
    passed = 0
    dj = None
    if judge == "dspy":
        if not _HAS_DSPY_JUDGE:
            raise RuntimeError("DSPy judge requested but dspy-ai not available")
        dj = DSPyJudge()

    for r in rules:
        q = r["q"]
        req_unver = bool(r.get("require_unverifiable", False))
        allow_paths = r.get("allow_paths")
        deny_paths = r.get("deny_paths")
        th = float(r.get("threshold", default_threshold))
        min_cites = int(r.get("min_citations", 1))
        ans = pc.ask(q, k=int(r.get("k", 5)), threshold=th, rerank=rerank, span_max_gap=span_max_gap, allowed_paths_regex=allow_paths, denied_paths_regex=deny_paths)
        compliant = False
        reason = ""
        if req_unver:
            compliant = ans.unverifiable
            reason = "required unverifiable" if compliant else "should be unverifiable"
        else:
            if not ans.unverifiable and len(ans.citations) >= min_cites:
                ok_src = True
                if allow_paths:
                    pat = re.compile(allow_paths)
                    ok_src = all(pat.search(c.path) for c in ans.citations)
                if deny_paths:
                    patd = re.compile(deny_paths)
                    ok_src = ok_src and all(not patd.search(c.path) for c in ans.citations)
                if ok_src and dj is not None:
                    ev_lines = [f"{c.path}:{c.line_no} | {c.text}" for c in ans.citations]
                    verdict = dj.judge(q, ev_lines)
                    v = verdict.get("verdict", "Unverifiable")
                    compliant = (v == "Compliant")
                    reason = f"judge={v}: {verdict.get('reason','')}"
                else:
                    compliant = ok_src
                    reason = "citations from allowed sources" if ok_src else "bad source"
            else:
                reason = "unverifiable or insufficient citations"
        if compliant:
            passed += 1
        results.append({
            "rule": r,
            "answer": ans.answer,
            "unverifiable": ans.unverifiable,
            "citations": [{"path": c.path, "line_no": c.line_no, "text": c.text, "score": c.score} for c in ans.citations],
            "spans": getattr(ans, "spans", []),
            "max_score": ans.max_score,
            "threshold": ans.threshold,
            "compliant": compliant,
            "reason": reason,
        })

    summary = {"total": len(rules), "passed": passed, "pass_rate": (passed / len(rules)) if rules else None}
    return {"summary": summary, "results": results}


def main():
    ap = argparse.ArgumentParser(description="Regulatory evaluation harness producing proof bundles.")
    ap.add_argument("--mode", choices=["baseline", "dspy"], default="baseline")
    ap.add_argument("--docs", required=True, help="Glob for documents")
    ap.add_argument("--rules", required=True, help="JSONL rules file")
    ap.add_argument("--rerank", choices=["none", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--span_max_gap", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--judge", choices=["none", "dspy"], default="none")
    args = ap.parse_args()

    bundle = run_regulatory_eval(args.mode, args.docs, args.rules, default_threshold=args.threshold, rerank=args.rerank, span_max_gap=args.span_max_gap, judge=(None if args.judge == "none" else args.judge))
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
