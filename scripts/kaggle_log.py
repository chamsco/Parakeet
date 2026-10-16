"""Fetch a Kaggle kernel's log through the v1 API, without the CLI.

The CLI authenticates by introspecting the token against api.kaggle.com and that call has been failing with
SSLEOFError from this machine, while the same token over the same API answers fine when called directly. So this is the
fallback: one request, the log printed, no dependency on an endpoint that is having a bad day.

    python scripts/kaggle_log.py <owner>/<kernel-slug>
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def token() -> str:
    for name in ("access_token", "kaggle.json"):
        path = Path.home() / ".kaggle" / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            payload = json.loads(text)
            value = str(payload.get("key") or payload.get("access_token") or "").strip()
            if value:
                return value
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            if line.strip().startswith("KAGGLE_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        if text and "=" not in text and " " not in text:
            return text
    raise SystemExit("no token found")


def main() -> int:
    if len(sys.argv) < 2 or "/" not in sys.argv[1]:
        print("usage: python scripts/kaggle_log.py <owner>/<kernel-slug>")
        return 2
    owner, slug = sys.argv[1].split("/", 1)
    for path in (f"/kernels/output?userName={owner}&kernelName={slug}",
                 f"/kernels/output/{owner}/{slug}"):
        request = urllib.request.Request("https://www.kaggle.com/api/v1" + path)
        request.add_header("Authorization", f"Bearer {token()}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode())
            print(f"  fetched via {path}")
            for line in (payload.get("log") or [])[-40:]:
                print(" ", str(line).rstrip()[:170])
            if payload.get("files"):
                print("  output files:", [f.get("name") or f for f in payload["files"]][:10])
            return 0
        except urllib.error.HTTPError as error:
            print(f"  {path} -> HTTP {error.code}")
        except Exception as error:  # noqa: BLE001
            print(f"  {path} -> {type(error).__name__}")
    print("  neither output form worked; the kernel may still be running")
    return 1


if __name__ == "__main__":
    sys.exit(main())
