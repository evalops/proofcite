import json
import sys
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

def ask(q: str):
    req = urllib.request.Request(
        f"{API}/ask",
        data=json.dumps({"q": q, "k": 5, "threshold": 0.35}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data


if __name__ == "__main__":
    res = ask("What port does Jellyfin use?")
    print(json.dumps(res, indent=2))

