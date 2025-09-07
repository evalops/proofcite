import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proofcite.core import ProofCite
from proofcite.dspy_variant import ProofCiteDSPy


class DummyPred:
    def __init__(self, answer_json):
        self.answer_json = answer_json


class DummyProgram:
    def __init__(self, payload):
        self.payload = payload

    def __call__(self, question, evidence):
        return DummyPred(self.payload)


def build_pc(tmp_path, lines):
    p = tmp_path / "doc.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    pc = ProofCite()
    pc.add_documents([str(p)])
    pc.build()
    return pc


def test_dspy_fallback_on_bad_json(tmp_path):
    pc = build_pc(tmp_path, ["alpha", "beta", "gamma"])
    dsp = ProofCiteDSPy()
    dsp.retriever = pc  # bypass build()
    dsp._program = DummyProgram("not json")
    ans = dsp.ask("beta", k=3, threshold=0.0)
    assert not ans.unverifiable
    # Should fall back to deterministic best line, not crash
    assert "beta" in ans.answer


def test_dspy_citation_whitelist(tmp_path):
    pc = build_pc(tmp_path, ["alpha", "beta", "gamma"])
    dsp = ProofCiteDSPy()
    dsp.retriever = pc
    # Try to cite a non-existent path/line; expect unverifiable due to empty out_cites
    payload = '{"answer":"some","citations":[{"path":"/nope.txt","line_no":99}]}'
    dsp._program = DummyProgram(payload)
    ans = dsp.ask("beta", k=3, threshold=0.0)
    assert ans.unverifiable

