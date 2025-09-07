from __future__ import annotations
import glob
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import ProofCite, Answer
from .config import load_settings

try:
    from .dspy_variant import ProofCiteDSPy  # type: ignore
    _HAS_DSPY = True
except Exception:
    ProofCiteDSPy = None  # type: ignore
    _HAS_DSPY = False


class AskRequest(BaseModel):
    q: str = Field(..., description="Question")
    k: int = Field(5, ge=1, le=50)
    threshold: float = Field(0.35, ge=0.0, le=1.0)
    rerank: Optional[str] = Field("none", description="none|bm25|hybrid")
    span_max_gap: Optional[int] = Field(0, description="Merge lines within this gap into spans")
    allow_paths: Optional[str] = Field(None, description="Regex of allowed citation paths")
    deny_paths: Optional[str] = Field(None, description="Regex of disallowed citation paths")


class AskResponse(BaseModel):
    answer: str
    unverifiable: bool
    max_score: float
    threshold: float
    citations: List[dict]
    spans: List[dict]


app = FastAPI(title="ProofCite API", version="0.1.0")


class _Service:
    def __init__(self):
        s = load_settings()
        self.mode = os.getenv("PROOFCITE_MODE", "baseline")  # baseline|dspy
        self.docs_glob = os.getenv("PROOFCITE_DOCS", s.docs)
        self.segment = s.segment
        self.token_chunk_size = s.token_chunk_size
        self.retriever_kind = os.getenv("PROOFCITE_RETRIEVER", "deterministic")  # deterministic|embedding
        self.pc: Optional[object] = None

    def build(self, docs_glob: Optional[str] = None, mode: Optional[str] = None, segment: Optional[str] = None, token_chunk_size: Optional[int] = None, retriever: Optional[str] = None):
        if docs_glob:
            self.docs_glob = docs_glob
        if mode:
            self.mode = mode
        if segment:
            self.segment = segment
        if token_chunk_size is not None:
            self.token_chunk_size = token_chunk_size
        if retriever:
            self.retriever_kind = retriever
        paths = sorted(glob.glob(self.docs_glob))
        if not paths:
            raise RuntimeError(f"No documents matched glob: {self.docs_glob}")
        if self.mode == "baseline":
            if self.retriever_kind == "deterministic":
                pc = ProofCite(segment=self.segment, token_chunk_size=self.token_chunk_size)
            elif self.retriever_kind == "embedding":
                from .embedding import EmbeddingRetriever
                pc = EmbeddingRetriever(segment=self.segment, token_chunk_size=self.token_chunk_size)
            else:
                raise RuntimeError("Invalid PROOFCITE_RETRIEVER: use 'deterministic' or 'embedding'")
        elif self.mode == "dspy":
            if not _HAS_DSPY:
                raise RuntimeError("DSPy mode requested but dspy-ai is not installed.")
            pc = ProofCiteDSPy()
        else:
            raise RuntimeError("Invalid PROOFCITE_MODE: use 'baseline' or 'dspy'")
        pc.add_documents(paths)
        pc.build()
        self.pc = pc

    def ask(self, q: str, k: int, threshold: float, rerank: str = "none", span_max_gap: int = 0, allow_paths: Optional[str] = None, deny_paths: Optional[str] = None) -> Answer:
        if self.pc is None:
            self.build()
        return self.pc.ask(q, k=k, threshold=threshold, rerank=rerank, span_max_gap=span_max_gap, allowed_paths_regex=allow_paths, denied_paths_regex=deny_paths)  # type: ignore


service = _Service()


@app.on_event("startup")
def _on_start():
    try:
        service.build()
    except Exception as e:
        # Defer error to first /ask if no docs available at startup
        print(f"[startup] build skipped: {e}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "mode": service.mode,
        "docs_glob": service.docs_glob,
        "has_index": service.pc is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        ans = service.ask(req.q, k=req.k, threshold=req.threshold, rerank=(req.rerank or "none"), span_max_gap=(req.span_max_gap or 0), allow_paths=req.allow_paths, deny_paths=req.deny_paths)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AskResponse(
        answer=ans.answer,
        unverifiable=ans.unverifiable,
        max_score=ans.max_score,
        threshold=ans.threshold,
        citations=[
            {"path": c.path, "line_no": c.line_no, "text": c.text, "score": c.score}
            for c in ans.citations
        ],
        spans=getattr(ans, 'spans', []),
    )


class BatchAskRequest(BaseModel):
    qs: List[str]
    k: int = Field(5, ge=1, le=50)
    threshold: float = Field(0.35, ge=0.0, le=1.0)
    rerank: Optional[str] = Field("none", description="none|bm25|hybrid")
    span_max_gap: Optional[int] = Field(0, description="Merge lines within this gap into spans")
    allow_paths: Optional[str] = Field(None, description="Regex of allowed citation paths")
    deny_paths: Optional[str] = Field(None, description="Regex of disallowed citation paths")


@app.post("/batch")
def batch(req: BatchAskRequest):
    out = []
    for q in req.qs:
        try:
            ans = service.ask(q, k=req.k, threshold=req.threshold, rerank=(req.rerank or "none"), span_max_gap=(req.span_max_gap or 0), allow_paths=req.allow_paths, deny_paths=req.deny_paths)
            out.append({
                "q": q,
                "answer": ans.answer,
                "unverifiable": ans.unverifiable,
                "max_score": ans.max_score,
                "threshold": ans.threshold,
                "citations": [
                    {"path": c.path, "line_no": c.line_no, "text": c.text, "score": c.score}
                    for c in ans.citations
                ],
                "spans": getattr(ans, 'spans', []),
            })
        except Exception as e:
            out.append({"q": q, "error": str(e)})
    return {"results": out}


class ReloadRequest(BaseModel):
    docs: Optional[str] = Field(None, description="Glob for docs")
    mode: Optional[str] = Field(None, description="baseline or dspy")
    segment: Optional[str] = Field(None, description="line|paragraph|sentence|token")
    token_chunk_size: Optional[int] = Field(None, description="token chunk size when segment=token")
    retriever: Optional[str] = Field(None, description="deterministic|embedding")


@app.post("/reload")
def reload(req: ReloadRequest):
    try:
        service.build(docs_glob=req.docs, mode=req.mode, segment=req.segment, token_chunk_size=req.token_chunk_size, retriever=req.retriever)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "mode": service.mode, "docs_glob": service.docs_glob, "segment": service.segment, "token_chunk_size": service.token_chunk_size, "retriever": service.retriever_kind}


def main():  # console script entry
    import uvicorn
    uvicorn.run("proofcite.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
