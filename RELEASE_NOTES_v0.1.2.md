# ProofCite v0.1.2 — Hardened Core, Evals Scaffold, Pluggable Retrieval

Highlights
- Tests + CI: Added pytest unit tests (core retrieval, spans, filters) and a GitHub Actions workflow.
- Config: Pydantic `Settings` (`PROOFCITE_` env) for defaults — threshold, rerank, span gap, segmentation, docs.
- Segmentation: Select `line|paragraph|sentence|token` and token chunk size for flexible ingestion.
- Pluggable retrievers: `DeterministicRetriever` (TF‑IDF/BM25 core) and a lightweight `EmbeddingRetriever` (Hashing + cosine).
- Evals framework scaffold: Pydantic models (Requirement, Proof, Assessment), runner, and assessment engine.
- UI/API: Gradio gets a “Regulatory Proof” tab and retriever/segmentation controls; API `/reload` now accepts `segment/token_chunk_size/retriever`.

Why it matters
- Reliability: Formal tests + fail‑closed retrieval build trust (especially for regulated use cases).
- Extensibility: Pluggable retriever and evidence model unlock future embedding/DB retrievers without changing the surface.
- Proof‑first evals: The `proofcite.evals` scaffold makes it straightforward to convert regulations into Requirements and produce audit‑ready Assessments.

Upgrade notes
- Install deps: `pip install -r proofcite/requirements.txt`
- Segmentation can lower per‑chunk scores; adjust `--threshold` accordingly (e.g., `--segment sentence --threshold 0.2`).
- Docker publish workflow will push on tags `vX.Y.Z` to DockerHub if `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are set in repo secrets.

Try it
- Value report: `python -m proofcite.examples.report_value`
- Regulatory harness: `python -m proofcite.regulatory --mode baseline --docs "proofcite/examples/regulatory/*.txt" --rules proofcite/examples/regulatory/rules_extended.jsonl --rerank hybrid --span_max_gap 1`
- UI: `proofcite-gradio` (set `PROOFCITE_DOCS` or change glob in the field, pick retriever and segmentation).

