from __future__ import annotations
import argparse
import glob
import json
from typing import List

from .models import Requirement
from .runner import RequirementRunner
from .assessor import AssessmentEngine
from ..retriever import DeterministicRetriever


def load_requirements(path: str) -> List[Requirement]:
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reqs.append(Requirement.model_validate_json(line))
    return reqs


def main():
    ap = argparse.ArgumentParser(description="Run requirements against a corpus and produce assessments.")
    ap.add_argument("--docs", required=True, help="Glob for documents")
    ap.add_argument("--requirements", required=True, help="JSONL requirements file")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.docs))
    r = DeterministicRetriever()
    r.add_documents(paths)
    r.build()

    reqs = load_requirements(args.requirements)
    proofs = RequirementRunner(r).run(reqs)
    assessments = AssessmentEngine().assess(reqs, proofs)
    out = {"assessments": [a.model_dump() for a in assessments]}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

