from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .core import Citation, Answer

_LINE_SPLIT = re.compile(r"\r?\n")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingRetriever:
    """Lightweight embedding-like retriever (HashingVectorizer + cosine).

    This serves as a pluggable alternative to the deterministic TF‑IDF retriever
    without adding heavy dependencies. Not meant to outperform TF‑IDF; this is a
    scaffold for the retriever interface.
    """

    def __init__(self, lowercase: bool = True, n_features: int = 4096, segment: str = "line", token_chunk_size: int = 80):
        self.lowercase = lowercase
        self.segment = segment
        self.token_chunk_size = token_chunk_size
        self.vec = HashingVectorizer(lowercase=lowercase, n_features=n_features, alternate_sign=False, token_pattern=r"[A-Za-z0-9_]+")
        self._fit = False
        self._X = None
        self._lines: List[str] = []
        self._meta: List[Tuple[str, int]] = []  # (path, line_no)

    def _segment_text(self, text: str) -> List[str]:
        mode = (self.segment or "line").lower()
        if mode == "line":
            parts = [ln.strip() for ln in _LINE_SPLIT.split(text)]
            return [p for p in parts if p]
        if mode == "paragraph":
            paras: List[str] = []
            buf: List[str] = []
            for ln in _LINE_SPLIT.split(text):
                if ln.strip():
                    buf.append(ln.strip())
                else:
                    if buf:
                        paras.append(" ".join(buf))
                        buf = []
            if buf:
                paras.append(" ".join(buf))
            return paras
        if mode == "sentence":
            parts = re.split(r"(?<=[.!?])\s+", text)
            return [p.strip() for p in parts if p and p.strip()]
        if mode == "token":
            toks = _TOKEN.findall(text)
            size = max(1, int(self.token_chunk_size or 80))
            return [" ".join(toks[i:i+size]) for i in range(0, len(toks), size)]
        return [text.strip()] if text.strip() else []

    def add_documents(self, paths: Iterable[str]) -> None:
        for p in paths:
            path = Path(p)
            text = path.read_text(encoding="utf-8", errors="ignore")
            segs = self._segment_text(text)
            for i, seg in enumerate(segs):
                self._lines.append(seg)
                self._meta.append((str(path), i + 1))

    def build(self) -> None:
        if not self._lines:
            raise ValueError("No documents loaded")
        self._X = self.vec.transform(self._lines)
        self._fit = True

    def _merge_spans(self, cites: List[Citation], max_gap: int = 0) -> List[Dict]:
        if max_gap <= 0 or not cites:
            return []
        out: List[Dict] = []
        by_path: Dict[str, List[Citation]] = {}
        for c in cites:
            if c.score <= 0:
                continue
            by_path.setdefault(c.path, []).append(c)
        for path, items in by_path.items():
            items.sort(key=lambda x: x.line_no)
            cur_start = items[0].line_no
            cur_end = items[0].line_no
            cur_texts = [items[0].text]
            cur_score = items[0].score
            for c in items[1:]:
                if c.line_no <= cur_end + max_gap:
                    if c.line_no > cur_end:
                        cur_end = c.line_no
                        cur_texts.append(c.text)
                    cur_score = max(cur_score, c.score)
                else:
                    out.append({"path": path, "start_line": cur_start, "end_line": cur_end, "text": " ".join(cur_texts), "score": float(cur_score)})
                    cur_start = c.line_no
                    cur_end = c.line_no
                    cur_texts = [c.text]
                    cur_score = c.score
            out.append({"path": path, "start_line": cur_start, "end_line": cur_end, "text": " ".join(cur_texts), "score": float(cur_score)})
        out.sort(key=lambda d: -d["score"])  # best first
        return out

    def ask(self, q: str, k: int = 5, threshold: float = 0.35, rerank: str = "none", span_max_gap: int = 0, allowed_paths_regex: Optional[str] = None, denied_paths_regex: Optional[str] = None) -> Answer:
        if not self._fit:
            raise RuntimeError("Index not built; call build()")
        qv = self.vec.transform([q])
        sims = cosine_similarity(qv, self._X)[0]
        # Path filter
        allow_re = re.compile(allowed_paths_regex) if allowed_paths_regex else None
        deny_re = re.compile(denied_paths_regex) if denied_paths_regex else None
        eligible = np.ones_like(sims, dtype=bool)
        if allow_re or deny_re:
            for i, (p, _) in enumerate(self._meta):
                if allow_re and not allow_re.search(p):
                    eligible[i] = False
                if deny_re and deny_re.search(p):
                    eligible[i] = False
        masked = sims.copy(); masked[~eligible] = -1.0
        max_sim = float(np.max(masked)) if masked.size else 0.0
        if max_sim < threshold:
            return Answer(answer="Unverifiable", citations=[], spans=[], max_score=max_sim, threshold=threshold, unverifiable=True)
        ordered = np.argsort(-masked)[:k]
        cites: List[Citation] = []
        for idx in ordered:
            path, line_no = self._meta[int(idx)]
            cites.append(Citation(path=path, line_no=line_no, text=self._lines[int(idx)], score=float(sims[int(idx)])))
        spans = self._merge_spans(cites, max_gap=span_max_gap)
        best_text = spans[0]["text"] if spans else cites[0].text
        return Answer(answer=best_text, citations=cites, spans=spans, max_score=max_sim, threshold=threshold, unverifiable=False)

