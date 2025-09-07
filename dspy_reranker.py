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


class ScoreLines(dspy.Signature if _HAS_DSPY else object):  # type: ignore
    """Score evidence lines for relevance to the question.

    Return STRICT JSON list of floats between 0 and 1 with one score per line.
    e.g., {"scores": [0.9, 0.2, 0.1]}
    """
    question: str = dspy.InputField() if _HAS_DSPY else None  # type: ignore
    lines: str = dspy.InputField() if _HAS_DSPY else None  # type: ignore
    result_json: str = dspy.OutputField(desc="Strict JSON: {scores: [...]} ") if _HAS_DSPY else None  # type: ignore


class DSPyReranker:
    def __init__(self, model: Optional[str] = None):
        _ensure(model=model)
        self._predict = dspy.Predict(ScoreLines)

    def rerank(self, question: str, lines: List[str]) -> List[float]:
        pred = self._predict(question=question, lines="\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines)))
        try:
            data = json.loads((pred.result_json or "").strip())  # type: ignore
            scores = data.get("scores", [])
            if not isinstance(scores, list) or len(scores) != len(lines):
                raise ValueError("bad scores length")
            return [float(max(0.0, min(1.0, s))) for s in scores]
        except Exception:
            # Fallback to equal scoring
            return [0.5] * len(lines)

