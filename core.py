from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
from pathlib import Path
import re
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_LINE_SPLIT = re.compile(r"\r?\n")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")

@dataclass
class Citation:
    path: str
    line_no: int
    text: str
    score: float

@dataclass
class Answer:
    answer: str
    citations: List[Citation]
    spans: List[Dict]
    max_score: float
    threshold: float
    unverifiable: bool

class ProofCite:
    def __init__(self, lowercase: bool = True, ngram_max: int = 2, min_df: int = 1,
                 segment: str = "line", token_chunk_size: int = 80):
        self.lowercase = lowercase
        self.vectorizer = TfidfVectorizer(lowercase=lowercase, ngram_range=(1, ngram_max), min_df=min_df)
        self._fit = False
        self._X = None
        self._lines: List[str] = []
        self._meta: List[Tuple[str, int]] = []  # (path, line_no)
        self.segment = segment
        self.token_chunk_size = token_chunk_size
        # For BM25
        self._tokens: List[List[str]] = []
        self._df: Dict[str, int] = {}
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0

    def _segment_text(self, text: str) -> List[str]:
        mode = (self.segment or "line").lower()
        if mode == "line":
            parts = [ln.strip() for ln in _LINE_SPLIT.split(text)]
            return [p for p in parts if p]
        if mode == "paragraph":
            # Split on blank lines
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
            # Naive sentence split
            parts = re.split(r"(?<=[.!?])\s+", text)
            return [p.strip() for p in parts if p and p.strip()]
        if mode == "token":
            toks = _TOKEN.findall(text)
            size = max(1, int(self.token_chunk_size or 80))
            out = []
            for i in range(0, len(toks), size):
                out.append(" ".join(toks[i:i+size]))
            return out
        # Fallback
        return [text.strip()] if text.strip() else []

    def add_documents(self, paths: Iterable[str]) -> None:
        for p in paths:
            path = Path(p)
            suffix = path.suffix.lower()
            # JSON Lines: each record has a text/content/body field
            if suffix == ".jsonl":
                try:
                    with path.open("r", encoding="utf-8") as f:
                        for rec_idx, raw in enumerate(f, start=1):
                            raw = raw.strip()
                            if not raw:
                                continue
                            try:
                                obj = json.loads(raw)
                            except Exception:
                                continue
                            text = None
                            for key in ("text", "content", "body", "abstract"):
                                v = obj.get(key)
                                if isinstance(v, str):
                                    text = v
                                    break
                            if not text:
                                continue
                            segs = self._segment_text(text)
                            for i, seg in enumerate(segs):
                                self._lines.append(seg)
                                self._meta.append((f"{str(path)}@rec={rec_idx}", i + 1))
                    continue
                except Exception:
                    # fall through to raw read
                    pass
            # CSV: concatenate row or take text/content column
            if suffix == ".csv":
                try:
                    import csv
                    with path.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row_idx, row in enumerate(reader, start=1):
                            text = row.get("text") or row.get("content") or " ".join(str(v) for v in row.values() if v)
                            if not text:
                                continue
                            s = text.strip()
                            if not s:
                                continue
                            segs = self._segment_text(s)
                            for i, seg in enumerate(segs):
                                self._lines.append(seg)
                                self._meta.append((f"{str(path)}@row={row_idx}", i + 1))
                    continue
                except Exception:
                    pass
            # PDF via pypdf, optional
            if suffix == ".pdf":
                try:
                    from pypdf import PdfReader  # type: ignore
                    reader = PdfReader(str(path))
                    full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
                    segs = self._segment_text(full_text)
                    for i, seg in enumerate(segs):
                        self._lines.append(seg)
                        self._meta.append((str(path), i + 1))
                    continue
                except Exception:
                    pass
            # Default: raw text
            text = path.read_text(encoding="utf-8", errors="ignore")
            segs = self._segment_text(text)
            for i, seg in enumerate(segs):
                self._lines.append(seg)
                self._meta.append((str(path), i + 1))

    def build(self) -> None:
        if not self._lines:
            raise ValueError("No documents loaded")
        self._X = self.vectorizer.fit_transform(self._lines)
        # Build BM25 structures
        self._tokens = []
        self._df = {}
        self._doc_len = []
        for line in self._lines:
            s = line.lower() if self.lowercase else line
            toks = _TOKEN.findall(s)
            self._tokens.append(toks)
            self._doc_len.append(len(toks))
            seen = set(toks)
            for t in seen:
                self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = float(np.mean(self._doc_len)) if self._doc_len else 0.0
        self._fit = True

    def _bm25_scores(self, q: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        s = q.lower() if self.lowercase else q
        q_terms = _TOKEN.findall(s)
        N = len(self._lines)
        if N == 0 or not q_terms:
            return np.zeros(N)
        idf: Dict[str, float] = {}
        for term in set(q_terms):
            df = self._df.get(term, 0)
            idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        scores = np.zeros(N, dtype=np.float32)
        for i, toks in enumerate(self._tokens):
            dl = self._doc_len[i] if i < len(self._doc_len) else 0
            if dl == 0:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            denom_const = k1 * (1 - b + b * (dl / (self._avgdl or 1.0)))
            score = 0.0
            for term in q_terms:
                if term not in idf:
                    continue
                f = tf.get(term, 0)
                if f == 0:
                    continue
                score += idf[term] * ((f * (k1 + 1.0)) / (f + denom_const))
            scores[i] = score
        return scores

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
                    out.append({
                        "path": path,
                        "start_line": cur_start,
                        "end_line": cur_end,
                        "text": " ".join(cur_texts),
                        "score": float(cur_score),
                    })
                    cur_start = c.line_no
                    cur_end = c.line_no
                    cur_texts = [c.text]
                    cur_score = c.score
            out.append({
                "path": path,
                "start_line": cur_start,
                "end_line": cur_end,
                "text": " ".join(cur_texts),
                "score": float(cur_score),
            })
        out.sort(key=lambda d: -d["score"])  # best first
        return out

    def ask(self, q: str, k: int = 5, threshold: float = 0.35, rerank: str = "none", span_max_gap: int = 0, allowed_paths_regex: Optional[str] = None, denied_paths_regex: Optional[str] = None) -> Answer:
        if not self._fit:
            raise RuntimeError("Index not built; call build()")
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self._X)[0]
        # Path-based eligibility
        allow_re = re.compile(allowed_paths_regex) if allowed_paths_regex else None
        deny_re = re.compile(denied_paths_regex) if denied_paths_regex else None
        eligible = np.ones_like(sims, dtype=bool)
        if allow_re or deny_re:
            paths = [m[0] for m in self._meta]
            for i, p in enumerate(paths):
                if allow_re and not allow_re.search(p):
                    eligible[i] = False
                if deny_re and deny_re.search(p):
                    eligible[i] = False
        if not np.any(eligible):
            return Answer(answer="Unverifiable", citations=[], spans=[], max_score=0.0, threshold=threshold, unverifiable=True)
        masked = sims.copy()
        masked[~eligible] = -1.0
        max_sim = float(np.max(masked)) if masked.size else 0.0
        if max_sim < threshold:
            return Answer(answer="Unverifiable", citations=[], spans=[], max_score=max_sim, threshold=threshold, unverifiable=True)

        # Reranking
        if rerank in ("bm25", "hybrid"):
            bm25 = self._bm25_scores(q)
            bm25[~eligible] = -1.0
            pool_size = min(len(masked), max(k * 5, 50))
            pool = np.argsort(-masked)[:pool_size]
            if rerank == "bm25":
                order = sorted(pool.tolist(), key=lambda i: bm25[i], reverse=True)
                order = [i for i in order if eligible[i]]
                top_idx = np.array(order[:k])
            else:
                s = masked[pool]
                b = bm25[pool]
                def norm(x):
                    a = float(np.min(x))
                    z = float(np.max(x))
                    if z - a <= 1e-12:
                        return np.zeros_like(x)
                    return (x - a) / (z - a)
                combined = norm(s) + norm(b)
                ordered = [pool[i] for i in np.argsort(-combined)]
                ordered = [i for i in ordered if eligible[i]]
                top_idx = np.array(ordered[:k])
        elif rerank == "dspy":
            try:
                from .dspy_reranker import DSPyReranker  # type: ignore
                pool_size = min(len(masked), max(k * 5, 50))
                pool = [i for i in np.argsort(-masked)[:pool_size] if eligible[i]]
                lines = [self._lines[int(i)] for i in pool]
                scores = DSPyReranker().rerank(q, lines)
                order = [pool[i] for i in np.argsort(-np.array(scores))]
                top_idx = np.array(order[:k])
            except Exception:
                # Fallback to hybrid if DSPy reranker unavailable
                bm25 = self._bm25_scores(q)
                bm25[~eligible] = -1.0
                pool_size = min(len(masked), max(k * 5, 50))
                pool = np.argsort(-masked)[:pool_size]
                s = masked[pool]
                b = bm25[pool]
                def norm(x):
                    a = float(np.min(x)); z = float(np.max(x))
                    return np.zeros_like(x) if z-a <= 1e-12 else (x-a)/(z-a)
                combined = norm(s) + norm(b)
                ordered = [pool[i] for i in np.argsort(-combined)]
                ordered = [i for i in ordered if eligible[i]]
                top_idx = np.array(ordered[:k])
        else:
            ordered = np.argsort(-masked)
            ordered = [i for i in ordered if eligible[i]]
            top_idx = np.array(ordered[:k])
        cites: List[Citation] = []
        for idx in top_idx:
            path, line_no = self._meta[int(idx)]
            cites.append(Citation(path=path, line_no=line_no, text=self._lines[int(idx)], score=float(sims[int(idx)])))
        # Return top span if available
        spans = self._merge_spans(cites, max_gap=span_max_gap)
        if spans:
            best_text = spans[0]["text"]
        else:
            best_text = cites[0].text
        return Answer(answer=best_text, citations=cites, spans=spans, max_score=max_sim, threshold=threshold, unverifiable=False)

    def to_dict(self) -> Dict:
        return {
            "lowercase": self.lowercase,
            "lines": self._lines,
            "meta": self._meta,
            "vocabulary": self.vectorizer.vocabulary_ if self._fit else None,
            "avgdl": self._avgdl,
        }
