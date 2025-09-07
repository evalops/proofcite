from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EvidenceUnit:
    """A single, citable unit of evidence.

    source_id: stable identifier (e.g., filepath, record pointer)
    start_line/end_line: 1-based range within the source (for non-line units, keep equal)
    text: the textual content of the unit
    meta: extra metadata (e.g., scores, record indices)
    """
    source_id: str
    start_line: int
    end_line: int
    text: str
    meta: Dict[str, object] = field(default_factory=dict)


def unit_from_line(source_id: str, line_no: int, text: str, score: Optional[float] = None) -> EvidenceUnit:
    meta = {}
    if score is not None:
        meta["score"] = float(score)
    return EvidenceUnit(source_id=source_id, start_line=line_no, end_line=line_no, text=text, meta=meta)

