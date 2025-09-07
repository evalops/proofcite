from __future__ import annotations
from typing import Iterable, Protocol, Optional

from .core import ProofCite, Answer


class Retriever(Protocol):
    def add_documents(self, paths: Iterable[str]) -> None: ...
    def build(self) -> None: ...
    def ask(
        self,
        q: str,
        k: int = 5,
        threshold: float = 0.35,
        rerank: str = "none",
        span_max_gap: int = 0,
        allowed_paths_regex: Optional[str] = None,
        denied_paths_regex: Optional[str] = None,
    ) -> Answer: ...


class DeterministicRetriever:
    """Pluggable wrapper around the deterministic ProofCite engine.

    Keeps a stable interface while allowing other retrievers to be swapped in.
    """

    def __init__(self, **kwargs):
        self.pc = ProofCite(**kwargs)

    def add_documents(self, paths: Iterable[str]) -> None:
        self.pc.add_documents(paths)

    def build(self) -> None:
        self.pc.build()

    def ask(
        self,
        q: str,
        k: int = 5,
        threshold: float = 0.35,
        rerank: str = "none",
        span_max_gap: int = 0,
        allowed_paths_regex: Optional[str] = None,
        denied_paths_regex: Optional[str] = None,
    ) -> Answer:
        return self.pc.ask(
            q,
            k=k,
            threshold=threshold,
            rerank=rerank,
            span_max_gap=span_max_gap,
            allowed_paths_regex=allowed_paths_regex,
            denied_paths_regex=denied_paths_regex,
        )

