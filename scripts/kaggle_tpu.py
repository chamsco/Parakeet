"""Push a Kaggle kernel through the v1 API and read its log back.

Why the raw API rather than the `kaggle` CLI: the CLI here is v2.2.4 and did not pick up the token from
KAGGLE_API_TOKEN, while a direct call with the same token authenticated fine. The endpoints below are the documented v1
ones, so this script is the whole Kaggle surface this workflow needs - push, status, output.

The token is read from ~/.kaggle/kaggle.json (the new-style single string) and never printed. Load it into
KAGGLE_API_TOKEN before running this file.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://www.kaggle.com/api/v1"


def token() -> str:
    """The token value, from the environment or from whichever Kaggle file has it.

    `~/.kaggle/access_token` is checked before `kaggle.json` because Kaggle's settings page hands out a block of shell
    assignments ("KAGGLE_API_TOKEN=..." on one line and a second variable on the next), and a file holding that block
    authenticates nothing - the first attempt at this read the block verbatim and the API rejected the header. Both
    shapes are handled here so neither can silently become the reason a push fails.
    """
    env = os.environ.get("KAGGLE_API_TOKEN")
    if env:
        return env.strip().split("=", 1)[-1].strip()
    for name in ("access_token", "kaggle.json"):
        path = Path.home() / ".kaggle" / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            payload = json.loads(text)
            return str(payload.get("key") or payload.get("access_token") or "").strip()
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            if line.strip().startswith("KAGGLE_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        if text and "=" not in text and " " not in text:
            return text
    raise SystemExit("no Kaggle token found in KAGGLE_API_TOKEN, ~/.kaggle/access_token or ~/.kaggle/kaggle.json")


def _headers(account: str | None, style: str) -> dict:
    tok = token()
    if style == "bearer":
        return {"Authorization": f"Bearer {tok}"}
    import base64

    blob = base64.b64encode(f"{account}:{tok}".encode()).decode()
    return {"Authorization": f"Basic {blob}"}


def call(path: str, body: dict | None = None, method: str | None = None, account: str | None = None) -> dict:
    """Call the v1 API, trying bearer auth and falling back to the legacy basic style.

    The read endpoints accept the new-style token as a bearer credential, but /kernels/push answered 401 to that while
    /datasets/list answered 200 to the same token - so the write endpoints are tried both ways rather than assumed.
    """
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    last: str = ""
    for style in (("bearer", "basic") if account else ("bearer",)):
        request = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
        for header, value in _headers(account, style).items():
            request.add_header(header, value)
        if data:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode()
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as error:
            last = f"{style} -> HTTP {error.code}: {error.read().decode()[:300]}"
            if error.code not in (401, 403):
                raise SystemExit(f"{method or 'GET'} {path} -> {last}") from error
    raise SystemExit(f"{method or 'GET'} {path} -> {last}")


def whoami() -> str:
    """Derive the account name from a listing rather than asking for a whoami endpoint that the v1 API lacks."""
    for path in ("/datasets/list?maxSize=5", "/kernels/list?pageSize=5"):
        rows = call(path)
        for row in rows if isinstance(rows, list) else []:
            ref = row.get("ref") or ""
            if "/" in ref:
                return ref.split("/")[0]
    raise SystemExit("could not determine the account name from any listing")


def main() -> int:
    user = whoami()
    print(f"  authenticated as: {user}")
    slug = "parakeet-tpu-smoke"
    code = (ROOT / "kaggle" / "tpu_smoke.py").read_text(encoding="utf-8")
    body = {
        "slug": f"{user}/{slug}",
        "newTitle": "parakeet tpu smoke",
        "text": code,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": True,
        "enableInternet": True,
        "datasetDataSources": [],
        "kernelDataSources": [],
        "competitionDataSources": [],
    }
    print("  pushing with enableTpu=true ...")
    result = call("/kernels/push", body, account=user)
    print(f"  push response: {json.dumps(result)[:300]}")

    for attempt in range(60):
        time.sleep(20)
        status = call(f"/kernels/status?userName={user}&kernelName={slug}")
        state = status.get("status", "?")
        print(f"  [{attempt:02d}] status {state}")
        if state in ("complete", "error", "cancelled"):
            break
    output = call(f"/kernels/output?userName={user}&kernelName={slug}")
    print("  --- kernel log ---")
    for line in (output.get("log") or [])[-60:]:
        print(f"  {line.rstrip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
