from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Requirement(BaseModel):
    id: str
    text: str
    allow_paths: Optional[str] = Field(None, description="Regex for allowed sources")
    deny_paths: Optional[str] = Field(None, description="Regex for disallowed sources")
    min_citations: int = 1
    threshold: float = 0.35
    k: int = 5
    notes: Optional[str] = None


class EvidenceRef(BaseModel):
    path: str
    start_line: int
    end_line: int
    text: str
    score: float


class Proof(BaseModel):
    requirement_id: str
    answer: str
    unverifiable: bool
    evidence: List[EvidenceRef] = []
    max_score: float
    threshold: float


class Verdict(str, Enum):
    Compliant = "Compliant"
    NonCompliant = "NonCompliant"
    Contradictory = "Contradictory"
    Unverifiable = "Unverifiable"


class Assessment(BaseModel):
    requirement_id: str
    verdict: Verdict
    reason: str
    proof: Proof

