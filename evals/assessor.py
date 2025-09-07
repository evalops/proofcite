from __future__ import annotations
from typing import List

from .models import Requirement, Proof, Assessment, Verdict


class AssessmentEngine:
    def assess(self, reqs: List[Requirement], proofs: List[Proof]) -> List[Assessment]:
        by_id = {p.requirement_id: p for p in proofs}
        out: List[Assessment] = []
        for r in reqs:
            p = by_id.get(r.id)
            if p is None:
                out.append(Assessment(requirement_id=r.id, verdict=Verdict.Unverifiable, reason="no proof generated", proof=Proof(requirement_id=r.id, answer="", unverifiable=True, evidence=[], max_score=0.0, threshold=r.threshold)))
                continue
            if p.unverifiable:
                out.append(Assessment(requirement_id=r.id, verdict=Verdict.Unverifiable, reason="unverifiable", proof=p))
                continue
            # Simple contradiction heuristic: negative requirement with positive evidence or vice versa
            req_text = (r.text or "").lower()
            ans_text = (p.answer or "").lower()
            neg_cues = ["must not", "do not", "shall not", "prohibited"]
            pos_cues = ["must", "shall", "is required"]
            req_neg = any(c in req_text for c in neg_cues)
            req_pos = any(c in req_text for c in pos_cues) and not req_neg
            ans_neg = any(c in ans_text for c in neg_cues)
            ans_pos = any(c in ans_text for c in pos_cues) and not ans_neg
            if (req_neg and ans_pos) or (req_pos and ans_neg):
                out.append(Assessment(requirement_id=r.id, verdict=Verdict.Contradictory, reason="contradictory cues", proof=p))
                continue
            # Minimal policy: require min citations and assume compliant if present
            if len(p.evidence) >= r.min_citations:
                out.append(Assessment(requirement_id=r.id, verdict=Verdict.Compliant, reason=f">= {r.min_citations} citations", proof=p))
            else:
                out.append(Assessment(requirement_id=r.id, verdict=Verdict.NonCompliant, reason="insufficient citations", proof=p))
        return out
