# Changelog

## v0.1.0 – ProofCite initial release

- Deterministic TF‑IDF line‑level retrieval with fail‑closed thresholding.
- Line‑anchored citations and extractive answers.
- Typer CLI (`proofcite`) with `--json` output.
- Gradio demo (`proofcite-gradio`) with `PROOFCITE_DOCS` env support.
- Optional DSPy variant (`proofcite-dspy`) for LLM‑stitched, quote‑only answers.
- Eval harness: tiny devset + evaluator script.
- DSPy optimizer sketch for few‑shot compilation.
- FastAPI microservice (`proofcite-api`) exposing `/health`, `/ask`, `/reload`.
- Dockerfile and docker-compose for API + Gradio services.
- v0.1.2 – Tests, config, segmentation, retriever, evals scaffold
  - Added pytest unit tests and CI workflow (GitHub Actions).
  - Introduced Pydantic `Settings` (env-driven defaults) and segmentation modes: line/paragraph/sentence/token.
  - Pluggable retrievers: `DeterministicRetriever` (TF‑IDF/BM25 core) and a lightweight `EmbeddingRetriever` stub.
  - Regulatory eval scaffolding (`proofcite.evals`): data models, runner, assessment engine.
  - API/CLI support for retriever selection, constraints, reranker, and spans in responses.
