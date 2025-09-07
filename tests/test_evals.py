import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proofcite.retriever import DeterministicRetriever
from proofcite.evals.models import Requirement
from proofcite.evals.runner import RequirementRunner
from proofcite.evals.assessor import AssessmentEngine, Verdict


def test_requirement_runner_and_assessor(tmp_path):
    # Create a tiny corpus
    p = tmp_path / "rules.txt"
    p.write_text("Do not recommend off-label drug use.\n", encoding="utf-8")
    r = DeterministicRetriever()
    r.add_documents([str(p)])
    r.build()

    reqs = [Requirement(id="r1", text="Should the agent recommend off-label use of Drug X?", allow_paths="rules", min_citations=1, threshold=0.0)]
    runner = RequirementRunner(r)
    proofs = runner.run(reqs)
    assert len(proofs) == 1 and not proofs[0].unverifiable

    engine = AssessmentEngine()
    assessments = engine.assess(reqs, proofs)
    assert assessments[0].verdict == Verdict.Compliant

