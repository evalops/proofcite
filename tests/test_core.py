import sys, os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2]) if Path(__file__).resolve().parents else os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from proofcite.core import ProofCite


def build_small_pc(lines_by_file):
    # Create temp files in memory by writing to examples path (reuse existing ones)
    pc = ProofCite()
    pc.add_documents(lines_by_file)
    pc.build()
    return pc


def test_fail_closed_threshold(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("foo\nbar\n", encoding="utf-8")
    pc = ProofCite()
    pc.add_documents([str(p)])
    pc.build()
    ans = pc.ask("baz", threshold=0.99)
    assert ans.unverifiable


def test_merge_spans(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    pc = ProofCite()
    pc.add_documents([str(p)])
    pc.build()
    ans = pc.ask("beta", k=3, threshold=0.0, span_max_gap=1)
    # When only one relevant line, span is that line
    assert not ans.unverifiable
    assert ans.spans and ans.spans[0]["start_line"] == ans.spans[0]["end_line"] == 2


def test_bm25_scores_positive(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("jellyfin runs on 8096\nother text here\n", encoding="utf-8")
    pc = ProofCite()
    pc.add_documents([str(p)])
    pc.build()
    scores = pc._bm25_scores("jellyfin 8096")
    assert scores.max() > 0


def test_path_filters(tmp_path):
    p1 = tmp_path / "allowed.txt"
    p2 = tmp_path / "denied.txt"
    p1.write_text("alpha allowed source\n", encoding="utf-8")
    p2.write_text("alpha denied source\n", encoding="utf-8")
    pc = ProofCite()
    pc.add_documents([str(p1), str(p2)])
    pc.build()
    # Deny all but allowed
    ans = pc.ask("alpha", threshold=0.0, allowed_paths_regex="allowed", denied_paths_regex="denied")
    assert not ans.unverifiable
    assert all("allowed" in c.path for c in ans.citations)
