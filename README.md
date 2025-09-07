# Proof‑of‑Citation RAG — *Cite or shut up*

[![tests](https://github.com/evalops/proofcite/actions/workflows/tests.yml/badge.svg)](https://github.com/evalops/proofcite/actions/workflows/tests.yml)

**Why:** LLM answers without verifiable sources are liabilities. This returns only extractive answers with **line‑anchored citations**. If confidence is low, it **fails closed**.

**How it works (ASCII):**

```
question ─┐
          ├─> TF‑IDF over line‑level chunks ──> top‑k lines ──> threshold? ──> answer + [doc:line] cites
docs ─────┘                                                └──> else: "Unverifiable"
```

## Quick Start (≤6 commands)

```bash
# 1) create env
python -m venv .venv && . .venv/bin/activate
# 2) install
pip install -r requirements.txt
# 3) Regulatory harness (uses sample regulatory docs)
python -m proofcite.regulatory --mode baseline --docs "proofcite/examples/regulatory/*.txt" --rules proofcite/examples/regulatory/rules_min.jsonl --rerank hybrid --span_max_gap 1
# 4) CLI ask (regulatory question)
python -m proofcite.cli --docs "proofcite/examples/regulatory/*.txt" --q "Can we provide forward-looking investment advice?" --json
# 5) Gradio demo (HF Space compatible)
python -m proofcite.gradio_app
# 6) Docker 1‑click
docker build -t evalops/proofcite . && docker run --rm -p 7860:7860 evalops/proofcite
# 7) API server (FastAPI)
python -m proofcite.api  # serves on :8000
```

## What it can do

- **Direct extract** with **line anchors** over regulatory text *(~5–20 ms on CPU for small corpora)*
- **Fail closed** if cosine similarity below threshold *(default 0.35)*
- **Deterministic**: no API keys, runs anywhere
  
## Broadening Sources

- Supports `.txt/.md` line‑level citing by default.
- Also ingests `.jsonl` (uses `text`/`content`/`body`), `.csv` (row text), and optionally `.pdf` via `pypdf`.
- You can restrict citations to certain sources with `--allow-paths` (regex) and/or exclude with `--deny-paths`.

Examples:

```bash
# Ingest multiple regulatory sources and restrict cites
python -m proofcite.cli --docs "proofcite/examples/regulatory/*.txt" --q "Can we provide forward-looking investment advice?" --json \
  --rerank hybrid --span-max-gap 1 --allow-paths 'fda|sec|faa|hipaa|soc2'
```

## Install (pip)

```bash
pip install -e .[dspy]
# or without DSPy: pip install -e .
```

CLI entry points after install:

```bash
proofcite --docs "proofcite/examples/regulatory/*.txt" --q "Can we provide forward-looking investment advice?" --json
proofcite-dspy --docs "proofcite/examples/regulatory/*.txt" --q "Should the agent recommend off-label use of Drug X?" --json
proofcite-gradio  # launches UI on :7860
```

## DSPy (optional, LLM)

- Install: `pip install dspy-ai`
- Configure an LM via DSPy, e.g. `export OPENAI_API_KEY=...` and optionally `export DSPY_MODEL=openai/gpt-4o-mini`.
- Use CLI: `python -m proofcite.dspy_cli --docs "proofcite/examples/regulatory/*.txt" --q "Should the agent recommend off-label use of Drug X?" [--json]`.
- Batch mode: `python -m proofcite.dspy_cli --docs "proofcite/examples/regulatory/*.txt" --batch /path/to/questions.txt --json`
- Behavior: still fails closed via retrieval threshold. If above threshold, an LLM stitches a quote‑only answer and returns citations as JSON, enforcing extractive answers.

Ollama setup (local LLM via LiteLLM):

```bash
export DSPY_PROVIDER=ollama
export DSPY_MODEL=ollama/llama3    # or an installed Ollama model
export OLLAMA_BASE=http://localhost:11434
proofcite-dspy --docs "proofcite/examples/regulatory/*.txt" --q "Should the agent recommend off-label use of Drug X?" --json
```

## Adopt In Your Agent

- Python (deterministic):
  - `from proofcite.core import ProofCite`
  - `pc.add_documents([...]); pc.build(); ans = pc.ask("...", threshold=0.35)`
  - Check `ans.unverifiable`; otherwise use `ans.answer` and `ans.citations`.
- CLI JSON (easy integration):
  - `proofcite --docs "..." --q "..." --json | jq .`
  - Fields: `answer`, `unverifiable`, `max_score`, `threshold`, `citations[]`.
- DSPy judge (regulatory): use the Regulatory Proof tab with “Use DSPy Judge” or invoke `--judge dspy` in `proofcite.regulatory`.

## Regulatory Evals

- Harness: `python -m proofcite.regulatory --mode baseline --docs "proofcite/examples/regulatory/*.txt" --rules proofcite/examples/regulatory/rules_min.jsonl --rerank hybrid --span_max_gap 1`
- DSPy Judge: add `--judge dspy` (requires DSPy + model). The judge considers only presented evidence and returns a structured verdict.

### DSPy Judge Optimization (few-shot)

- Label a small set: `proofcite/examples/regulatory/judge_train.jsonl` (fields: requirement, evidence_lines[], verdict, reason).
- Compile demos: `python -m proofcite.evals.optimize_judge --train proofcite/examples/regulatory/judge_train.jsonl --out proofcite/evals/judge_demos.jsonl`
- Use at runtime: `export PROOFCITE_JUDGE_DEMOS=proofcite/evals/judge_demos.jsonl` then enable the judge (`--judge dspy` or UI toggle).

## Evals as Proof (Regulatory)

- Concept: Treat evaluations as a negotiation with evidence — produce a proof bundle instead of a single score.
- Harness: `python -m proofcite.regulatory --mode baseline --docs "proofcite/examples/regulatory/*.txt" --rules proofcite/examples/regulatory/rules_min.jsonl`
- Rules JSONL fields per line:
  - `q`: question; `require_unverifiable`: true → must fail closed
  - `allow_paths`/`deny_paths`: regex(es) constraining allowed evidence sources
  - `min_citations` (default 1), `threshold` (default 0.35)

Try:

```bash
python -m proofcite.regulatory --mode baseline \
  --docs "proofcite/examples/regulatory/*.txt" \
  --rules proofcite/examples/regulatory/rules_min.jsonl \
  --rerank hybrid --span_max_gap 1
```

## Run With Docker

- Build: `docker build -t evalops/proofcite .`
- Run: `docker run --rm -p 7860:7860 -e PROOFCITE_DOCS="proofcite/examples/regulatory/*.txt" evalops/proofcite`
  
### Docker Compose (API + Gradio)

```bash
docker compose up --build
# API:   http://localhost:8000
# Gradio:http://localhost:7860
```

### API Usage

```bash
curl -s localhost:8000/health | jq .
curl -s -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"q":"Can we provide forward-looking investment advice?","k":5,"threshold":0.35, "allow_paths":"sec"}' | jq .

# Batch
curl -s -X POST localhost:8000/batch \
  -H 'Content-Type: application/json' \
  -d '{"qs":["Should the agent recommend off-label use of Drug X?","Is PHI allowed in plaintext?"],"k":5,"threshold":0.35, "allow_paths":"fda|hipaa"}' | jq .
```

Client example: `python proofcite/examples/client.py`.

## Contact / CTA

- Add evaluations to your agent with EvalOps: https://evalops.dev/?utm_source=gh_proofcite_readme&utm_medium=referral&utm_campaign=readme_cta

## Changelog

See `CHANGELOG.md` (current: v0.1.2).

## Related work

- OpenAI "Retrieval augmented generation" primer
- BM25/TF‑IDF classical IR
- Line‑level citing in Elastic/ESQL style

## Roadmap

- Cross‑encoder re‑ranker (onnx, CPU‑friendly)
- Chunk merging for contiguous citations
- JSON Lines ingestion + embeddings option
