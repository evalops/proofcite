from __future__ import annotations
from typing import List, Optional
import json
import os

try:
    import dspy  # type: ignore
    _HAS_DSPY = True
except Exception:
    dspy = None  # type: ignore
    _HAS_DSPY = False


def _ensure(model: Optional[str] = None):
    if not _HAS_DSPY:
        raise RuntimeError("DSPy is not installed. pip install dspy-ai")
    if getattr(dspy.settings, "lm", None) is not None:
        return
    model = model or os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    provider = os.getenv("DSPY_PROVIDER", "")
    lm_kwargs = {}
    if provider.lower() == "ollama" or model.startswith("ollama/"):
        base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
        lm_kwargs.update({"api_base": base, "api_key": os.getenv("OLLAMA_API_KEY", "ollama")})
    dspy.configure(lm=dspy.LM(model=model, temperature=0.0, max_tokens=1000, **lm_kwargs))


class Judge(dspy.Signature if _HAS_DSPY else object):  # type: ignore
    """Given a requirement and evidence lines, output a structured verdict.

    Evidence must be treated as ground truth. If it contradicts the requirement,
    set verdict to "Contradictory". If it supports, set "Compliant". If
    insufficient, set "Unverifiable". Return STRICT JSON with keys:
    {"verdict": one of [Compliant, NonCompliant, Contradictory, Unverifiable],
     "reason": str}
    """
    requirement: str = dspy.InputField() if _HAS_DSPY else None  # type: ignore
    evidence: str = dspy.InputField() if _HAS_DSPY else None  # type: ignore
    result_json: str = dspy.OutputField(desc="Strict JSON verdict") if _HAS_DSPY else None  # type: ignore


class DSPyJudge:
    def __init__(self, model: Optional[str] = None):
        _ensure(model=model)
        self._predict = dspy.Predict(Judge)
        # Optionally prime with demos from JSONL if provided via env
        demos_path = os.getenv("PROOFCITE_JUDGE_DEMOS")
        if demos_path and os.path.exists(demos_path):
            try:
                demos = []
                with open(demos_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        demos.append(json.loads(line))
                # Attach demos directly if format matches DSPy's expected structure
                if demos:
                    self._predict.demos = demos  # type: ignore[attr-defined]
            except Exception:
                pass

    def judge(self, requirement_text: str, evidence_lines: List[str]) -> dict:
        pred = self._predict(requirement=requirement_text, evidence="\n".join(evidence_lines))
        try:
            data = json.loads((pred.result_json or "").strip())  # type: ignore
            v = str(data.get("verdict", "Unverifiable"))
            r = str(data.get("reason", ""))
            if v not in {"Compliant", "NonCompliant", "Contradictory", "Unverifiable"}:
                v = "Unverifiable"
            return {"verdict": v, "reason": r}
        except Exception:
            return {"verdict": "Unverifiable", "reason": "judge parse error"}
