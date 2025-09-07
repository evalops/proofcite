from __future__ import annotations
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Retrieval
    threshold: float = 0.35
    rerank: str = "none"  # none|bm25|hybrid
    span_max_gap: int = 0
    segment: str = "line"  # line|paragraph|sentence|token
    token_chunk_size: int = 80

    # Docs
    docs: str = "examples/data/*.txt"

    # DSPy / provider
    dspy_model: Optional[str] = None
    dspy_provider: Optional[str] = None
    ollama_base: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="PROOFCITE_", env_file=None, extra="ignore")


def load_settings() -> Settings:
    return Settings()  # reads from env

