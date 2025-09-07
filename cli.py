import glob
import json
import typer
from rich.console import Console
from .core import ProofCite
from .config import load_settings

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def ask(
    docs: str = typer.Option(..., "--docs", help="Glob for documents, e.g. 'examples/data/*.txt'"),
    q: str = typer.Option(None, "--q", help="Question"),
    batch: str = typer.Option(None, "--batch", help="Path to newline-delimited questions for batch mode"),
    k: int = typer.Option(5, help="Top-k lines to retrieve"),
    threshold: float = typer.Option(0.35, help="Minimum cosine similarity to answer"),
    rerank: str = typer.Option("none", help="Reranking: none|bm25|hybrid"),
    span_max_gap: int = typer.Option(0, help="Merge citations into spans when lines within this gap"),
    segment: str = typer.Option("line", help="Segmentation: line|paragraph|sentence|token"),
    token_chunk_size: int = typer.Option(80, help="Token chunk size when segment=token"),
    retriever: str = typer.Option("deterministic", help="Retriever: deterministic|embedding"),
    allow_paths: str = typer.Option(None, "--allow-paths", help="Regex of allowed citation paths"),
    deny_paths: str = typer.Option(None, "--deny-paths", help="Regex of disallowed citation paths"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON to stdout"),
):
    # Apply defaults from settings if CLI args not provided
    s = load_settings()
    docs = docs or s.docs
    threshold = threshold if threshold is not None else s.threshold
    rerank = rerank or s.rerank
    span_max_gap = span_max_gap if span_max_gap is not None else s.span_max_gap
    segment = segment or s.segment
    token_chunk_size = token_chunk_size or s.token_chunk_size

    paths = sorted(glob.glob(docs))
    if retriever == "deterministic":
        pc = ProofCite(segment=segment, token_chunk_size=token_chunk_size)
    elif retriever == "embedding":
        from .embedding import EmbeddingRetriever
        pc = EmbeddingRetriever(segment=segment, token_chunk_size=token_chunk_size)
    else:
        raise typer.BadParameter("retriever must be deterministic or embedding")
    pc.add_documents(paths)
    pc.build()
    if (q is None) == (batch is None):
        console.print("[bold red]Provide exactly one of --q or --batch[/]")
        raise typer.Exit(2)
    if batch is not None:
        with open(batch, "r", encoding="utf-8") as f:
            qs = [line.strip() for line in f if line.strip()]
        results = []
        for qi in qs:
            ans = pc.ask(qi, k=k, threshold=threshold, rerank=rerank, span_max_gap=span_max_gap, allowed_paths_regex=allow_paths, denied_paths_regex=deny_paths)
            results.append({
                "q": qi,
                "answer": ans.answer,
                "unverifiable": ans.unverifiable,
                "max_score": ans.max_score,
                "threshold": ans.threshold,
                "citations": [
                    {"path": c.path, "line_no": c.line_no, "text": c.text, "score": c.score}
                    for c in ans.citations
                ],
                "spans": ans.spans,
            })
        if json_out:
            print(json.dumps({"results": results}, ensure_ascii=False))
            # Return nonzero if any unverifiable
            if any(r["unverifiable"] for r in results):
                raise typer.Exit(1)
            return
        for r in results:
            if r["unverifiable"]:
                console.print(f"[bold red]Unverifiable[/] (q={r['q']})")
            else:
                console.print(f"[bold]Q:[/] {r['q']}\n[bold]A:[/] {r['answer']}")
                for c in r["citations"]:
                    console.print(f"  • {c['path']}:{c['line_no']}  score={c['score']:.3f}")
        # Exit 1 if any unverifiable
        if any(r["unverifiable"] for r in results):
            raise typer.Exit(1)
        return

    ans = pc.ask(q, k=k, threshold=threshold, rerank=rerank, span_max_gap=span_max_gap, allowed_paths_regex=allow_paths, denied_paths_regex=deny_paths)
    if json_out:
        payload = {
            "answer": ans.answer,
            "unverifiable": ans.unverifiable,
            "max_score": ans.max_score,
            "threshold": ans.threshold,
            "citations": [
                {"path": c.path, "line_no": c.line_no, "text": c.text, "score": c.score}
                for c in ans.citations
            ],
            "spans": ans.spans,
        }
        print(json.dumps(payload, ensure_ascii=False))
        if ans.unverifiable:
            raise typer.Exit(1)
        return
    if ans.unverifiable:
        console.print(f"[bold red]Unverifiable[/] (max_score={ans.max_score:.3f} < threshold={ans.threshold})")
        raise typer.Exit(1)
    console.print(f"[bold]Answer:[/] {ans.answer}")
    for c in ans.citations:
        console.print(f"  • {c.path}:{c.line_no}  score={c.score:.3f}")
    console.print(f"[dim]max_score={ans.max_score:.3f}  threshold={ans.threshold}[/]")

if __name__ == "__main__":
    app()

def main():  # console_script entry
    app()
