import glob
import os
import gradio as gr
from .core import ProofCite

def build_index(glob_pattern, retriever_kind="deterministic", segment="line", token_chunk_size=80):
    paths = sorted(glob.glob(glob_pattern))
    if retriever_kind == "embedding":
        from .embedding import EmbeddingRetriever
        pc = EmbeddingRetriever(segment=segment, token_chunk_size=token_chunk_size)
    else:
        pc = ProofCite(segment=segment, token_chunk_size=token_chunk_size)
    pc.add_documents(paths)
    pc.build()
    return pc

def make_app(default_glob: str | None = None):
    if default_glob is None:
        default_glob = os.getenv("PROOFCITE_DOCS", "examples/data/*.txt")
    retriever_kind = os.getenv("PROOFCITE_RETRIEVER", "deterministic")
    segment = os.getenv("PROOFCITE_SEGMENT", "line")
    token_chunk_size = int(os.getenv("PROOFCITE_TOKEN_CHUNK_SIZE", "80"))
    pc = build_index(default_glob, retriever_kind=retriever_kind, segment=segment, token_chunk_size=token_chunk_size)

    def rebuild(glob_pattern, retriever_kind, segment, token_size):
        nonlocal pc
        pc = build_index(glob_pattern, retriever_kind=retriever_kind, segment=segment, token_chunk_size=int(token_size))
        return f"Indexed {glob_pattern} ({retriever_kind}, {segment})"

    def answer(question, threshold, rerank, span_gap):
        ans = pc.ask(question, threshold=threshold, rerank=rerank, span_max_gap=int(span_gap))
        if ans.unverifiable:
            return "Unverifiable", ""
        cites_md = "\n".join([f"- `{c.path}:{c.line_no}` — {c.text}" for c in ans.citations])
        return ans.answer, cites_md

    def answer_reg(question, threshold, allow_paths, deny_paths, rerank, span_gap, use_dspy_judge):
        ans = pc.ask(question, threshold=threshold, rerank=rerank, span_max_gap=int(span_gap), allowed_paths_regex=allow_paths or None, denied_paths_regex=deny_paths or None)
        if ans.unverifiable:
            return "Unverifiable", ""
        spans_md = "\n".join([f"- `{s['path']}:{s['start_line']}-{s['end_line']}` — {s['text']}" for s in (ans.spans or [])])
        cites_md = "\n".join([f"- `{c.path}:{c.line_no}` — {c.text}" for c in ans.citations])
        verdict_md = ""
        if use_dspy_judge:
            try:
                from .evals.dspy_judge import DSPyJudge  # type: ignore
                dj = DSPyJudge()
                ev_lines = [f"{c.path}:{c.line_no} | {c.text}" for c in ans.citations]
                v = dj.judge(question, ev_lines)
                verdict_md = f"\n\nJudge: {v.get('verdict','Unverifiable')} — {v.get('reason','')}"
            except Exception as e:
                verdict_md = f"\n\nJudge error: {e}"
        return ans.answer + ("\n\nSpans:\n" + spans_md if spans_md else "") + verdict_md, cites_md

    with gr.Blocks() as demo:
        gr.Markdown("## Proof‑of‑Citation — cite or shut up")
        with gr.Row():
            glob_in = gr.Textbox(label="Docs Glob", value=default_glob)
            retriever_dd = gr.Dropdown(choices=["deterministic","embedding"], value=retriever_kind, label="Retriever")
            segment_dd = gr.Dropdown(choices=["line","paragraph","sentence","token"], value=segment, label="Segmentation")
            token_size = gr.Slider(minimum=20, maximum=400, step=10, value=token_chunk_size, label="Token chunk size")
        status = gr.Markdown()
        rebuild_btn = gr.Button("Rebuild Index")
        rebuild_btn.click(rebuild, inputs=[glob_in, retriever_dd, segment_dd, token_size], outputs=[status])
        with gr.Tab("Ask"):
            with gr.Row():
                q = gr.Textbox(label="Question", value="Should the agent recommend off-label use of Drug X?")
                th = gr.Slider(minimum=0.1, maximum=0.8, value=0.35, step=0.01, label="Confidence threshold")
            with gr.Row():
                rerank = gr.Dropdown(choices=["none","bm25","hybrid"], value="none", label="Rerank")
                span_gap = gr.Slider(minimum=0, maximum=3, value=0, step=1, label="Span merge gap")
            ans = gr.Textbox(label="Answer")
            cites = gr.Markdown(label="Citations (line‑anchored)")
            btn = gr.Button("Ask")
            btn.click(answer, inputs=[q, th, rerank, span_gap], outputs=[ans, cites])
        with gr.Tab("Regulatory Proof"):
            gr.Markdown("Constrain allowed sources and require line‑anchored proof.")
            with gr.Row():
                q2 = gr.Textbox(label="Question", value="Should the agent recommend off-label use of Drug X?")
                th2 = gr.Slider(minimum=0.1, maximum=0.8, value=0.35, step=0.01, label="Confidence threshold")
            with gr.Row():
                allow = gr.Textbox(label="Allow paths (regex)", value="fda|sec|faa|hipaa|soc2")
                deny = gr.Textbox(label="Deny paths (regex)")
            with gr.Row():
                rerank2 = gr.Dropdown(choices=["none","bm25","hybrid"], value="hybrid", label="Rerank")
                span_gap2 = gr.Slider(minimum=0, maximum=3, value=1, step=1, label="Span merge gap")
                use_judge = gr.Checkbox(label="Use DSPy Judge", value=False)
            ans2 = gr.Textbox(label="Answer + Spans")
            cites2 = gr.Markdown(label="Citations")
            btn2 = gr.Button("Prove")
            btn2.click(answer_reg, inputs=[q2, th2, allow, deny, rerank2, span_gap2, use_judge], outputs=[ans2, cites2])
    return demo

# Expose app at import time for HF Spaces
app = make_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

def main():  # console_script entry
    app.launch(server_name="0.0.0.0", server_port=7860)
