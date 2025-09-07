from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import json
import os

try:
    import dspy  # type: ignore
    _HAS_DSPY = True
except Exception:
    dspy = None  # type: ignore
    _HAS_DSPY = False

from .core import ProofCite, Citation, Answer


def _ensure_dspy_configured(model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 800) -> None:
    if not _HAS_DSPY:
        raise RuntimeError("DSPy is not installed. Please `pip install dspy-ai` and try again.")
    # If already configured, keep as-is.
    if getattr(dspy.settings, "lm", None) is not None:
        return
    model = model or os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    provider_name = os.getenv("DSPY_PROVIDER", "").lower()
    lm_kwargs = {}
    # Configure Ollama via LiteLLM (generic provider) by passing api_base.
    if provider_name == "ollama" or model.startswith("ollama/"):
        base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
        lm_kwargs.update({"api_base": base, "api_key": os.getenv("OLLAMA_API_KEY", "ollama")})
    # Users must provide provider credentials via env (e.g., OPENAI_API_KEY) as per DSPy docs.
    dspy.configure(lm=dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens, **lm_kwargs))


class GenerateAnswer(dspy.Signature if _HAS_DSPY else object):  # type: ignore
    """Craft an answer using only the provided evidence lines.

    Requirements:
    - Answer MUST be extractive: use verbatim quotes from the evidence only.
    - Include citations for each claim using the format {"path": str, "line_no": int}.
    - Output STRICT JSON with keys: answer (str), citations (list of {path, line_no}).
    - If evidence is insufficient, set answer to "Unverifiable" and citations to [].
    """
    question: str = dspy.InputField() if _HAS_DSPY else None  # type: ignore
    evidence: str = dspy.InputField() if _HAS_DSPY else None  # type: ignore
    answer_json: str = dspy.OutputField(desc="Strict JSON as specified above") if _HAS_DSPY else None  # type: ignore


class DSPyCiteProgram(dspy.Module if _HAS_DSPY else object):  # type: ignore
    def __init__(self):
        if not _HAS_DSPY:
            raise RuntimeError("DSPy is not installed. Please `pip install dspy-ai`." )
        super().__init__()
        self.gen = dspy.Predict(GenerateAnswer)

    def forward(self, question: str, evidence: str) -> dspy.Prediction:  # type: ignore
        return self.gen(question=question, evidence=evidence)


class ProofCiteDSPy:
    """Hybrid pipeline: TF-IDF retrieval + DSPy LLM answerer with citations.

    - Uses ProofCite for line-level retrieval and fail-closed thresholding.
    - Uses a DSPy program to stitch quotes into an answer with JSON citations.
    - Requires configuring DSPy with an LM provider (e.g., `DSPY_MODEL` env).
    """

    def __init__(self, lowercase: bool = True, ngram_max: int = 2, min_df: int = 1, model: Optional[str] = None):
        self.retriever = ProofCite(lowercase=lowercase, ngram_max=ngram_max, min_df=min_df)
        self._model = model
        self._program: Optional[DSPyCiteProgram] = None

    # Delegate ingestion/build to underlying retriever
    def add_documents(self, paths):
        self.retriever.add_documents(paths)

    def build(self):
        self.retriever.build()
        _ensure_dspy_configured(model=self._model)
        self._program = DSPyCiteProgram()

    def ask(self, q: str, k: int = 5, threshold: float = 0.35, rerank: str = "none", span_max_gap: int = 0, allowed_paths_regex: Optional[str] = None, denied_paths_regex: Optional[str] = None) -> Answer:
        if self._program is None:
            raise RuntimeError("Index not built; call build()")
        # Retrieve with deterministic retriever and enforce fail-closed
        base = self.retriever.ask(q, k=k, threshold=threshold, rerank=rerank, span_max_gap=span_max_gap, allowed_paths_regex=allowed_paths_regex, denied_paths_regex=denied_paths_regex)
        if base.unverifiable:
            return base

        # Format evidence: prefer spans if available, else lines
        if getattr(base, 'spans', None):
            evidence_lines = [
                f"{s['path']}:{s['start_line']}-{s['end_line']} | {s['text']}" for s in base.spans
            ]
        else:
            evidence_lines = [f"{c.path}:{c.line_no} | {c.text}" for c in base.citations]
        evidence_blob = "\n".join(evidence_lines)
        pred = self._program(question=q, evidence=evidence_blob)

        # Parse JSON output; fail safely to the best extractive line on errors
        try:
            data = json.loads((pred.answer_json or "").strip())  # type: ignore
            ans_text = data.get("answer", "Unverifiable")
            cites_in = data.get("citations", []) or []
        except Exception:
            # Fallback: keep deterministic best line and citations
            return base

        if not isinstance(cites_in, list):
            return base

        # Map returned citations back to retrieved lines (only allow those we retrieved)
        allowed = {(c.path, c.line_no): c for c in base.citations}
        out_cites: List[Citation] = []
        for item in cites_in:
            try:
                key = (str(item["path"]), int(item["line_no"]))
            except Exception:
                continue
            if key in allowed:
                out_cites.append(allowed[key])

        # Enforce fail-closed if the model claims unverifiable or returns no valid cites
        if (isinstance(ans_text, str) and ans_text.strip().lower() == "unverifiable") or not out_cites:
            return Answer(answer="Unverifiable", citations=[], spans=[], max_score=base.max_score, threshold=threshold, unverifiable=True)

        # Return model-crafted answer with validated citations (preserve spans from base)
        return Answer(answer=str(ans_text), citations=out_cites, spans=getattr(base, 'spans', []), max_score=base.max_score, threshold=threshold, unverifiable=False)
