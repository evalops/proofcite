from __future__ import annotations
from typing import List

from .models import Requirement, EvidenceRef, Proof
from ..retriever import Retriever


class RequirementRunner:
    def __init__(self, retriever: Retriever):
        self.r = retriever

    def run(self, reqs: List[Requirement]) -> List[Proof]:
        proofs: List[Proof] = []
        for r in reqs:
            ans = self.r.ask(
                r.text,
                k=r.k,
                threshold=r.threshold,
                rerank="hybrid",
                span_max_gap=1,
                allowed_paths_regex=r.allow_paths,
                denied_paths_regex=r.deny_paths,
            )
            ev = []
            if getattr(ans, "spans", None):
                for s in ans.spans:
                    ev.append(EvidenceRef(path=s["path"], start_line=int(s["start_line"]), end_line=int(s["end_line"]), text=str(s["text"]), score=float(s["score"])) )
            else:
                for c in ans.citations:
                    ev.append(EvidenceRef(path=c.path, start_line=c.line_no, end_line=c.line_no, text=c.text, score=c.score))
            proofs.append(Proof(requirement_id=r.id, answer=ans.answer, unverifiable=ans.unverifiable, evidence=ev, max_score=ans.max_score, threshold=ans.threshold))
        return proofs

