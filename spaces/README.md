# Hugging Face Space Setup

This repo already exposes a Gradio `app` in `proofcite/gradio_app.py` and reads the glob of documents from the `PROOFCITE_DOCS` environment variable.

Two options to publish a Space:

1) Minimal (use this repo as a Space)
- Create a new Space (SDK: Gradio, Hardware: CPU)
- Upload the `proofcite/` directory contents (or link to this repo)
- Set `app_file` to `proofcite/gradio_app.py` in the Space Settings
- Add `PROOFCITE_DOCS` secret (e.g., `proofcite/examples/data/*.txt`)

2) Stub app (copy-only)
- Create a Space and set `app.py` to:

    from proofcite.gradio_app import app

- Upload the `proofcite/` package directory alongside `app.py`.

Either approach will serve the UI at `/`.
