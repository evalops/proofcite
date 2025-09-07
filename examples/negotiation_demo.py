"""
Negotiation-as-proof demo: the auditor constrains allowed sources
and the system either produces grounded answers or fails closed.
"""

from proofcite.core import ProofCite
import glob

pc = ProofCite()
pc.add_documents(sorted(glob.glob("proofcite/examples/regulatory/*.txt")))
pc.build()

scenarios = [
    {"q": "Should the agent recommend off-label use of Drug X?", "allow": "fda_guidance"},
    {"q": "Can we provide forward-looking investment advice?", "allow": "sec_rules"},
]

for s in scenarios:
    ans = pc.ask(s["q"], k=5, threshold=0.35, rerank="hybrid", span_max_gap=1, allowed_paths_regex=s["allow"])
    print(f"Q: {s['q']}")
    if ans.unverifiable:
        print("  -> Unverifiable (insufficient evidence from allowed sources)")
    else:
        print(f"  -> {ans.answer}")
        for c in ans.citations[:3]:
            print(f"     cite: {c.path}:{c.line_no}")
    print()

