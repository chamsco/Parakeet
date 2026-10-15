"""Write only the token VALUE from Kaggle's clipboard block into the file the CLI reads.

Kaggle's settings page offers a block of shell assignments rather than a bare token, and pasting the block into
kaggle.json leaves a file whose first line is "KAGGLE_API_TOKEN=..." - which authenticates nothing.

This script reads that block, takes the value of KAGGLE_API_TOKEN, and writes it to ~/.kaggle/access_token, which is
where the CLI documents its token file. It NEVER prints the value, and it catches its own exceptions so that a failure
cannot echo the secret the way the previous attempt did.
"""

from __future__ import annotations

import pathlib
import sys
import urllib.request

HOME = pathlib.Path.home() / ".kaggle"
SRC = HOME / "kaggle.json"
DST = HOME / "access_token"


def extract() -> str | None:
    if not SRC.exists():
        print("  no kaggle.json to read")
        return None
    text = SRC.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("KAGGLE_API_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    # a bare token, which is the other legitimate shape
    single = text.strip()
    if single and "=" not in single and " " not in single:
        return single
    print("  kaggle.json has neither a KAGGLE_API_TOKEN= line nor a bare token")
    return None


def main() -> int:
    value = extract()
    if not value:
        return 1
    DST.write_text(value, encoding="ascii")
    print(f"  wrote access_token: {len(value)} characters, starting {value[:4]}... (rest withheld)")
    request = urllib.request.Request("https://www.kaggle.com/api/v1/datasets/list?maxSize=3")
    request.add_header("Authorization", f"Bearer {value}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
        print(f"  reads: HTTP {response.status}, {body.count('ref')} refs in the payload")
    except Exception as error:  # redacted on purpose: the message of a header error echoes the token
        print(f"  reads failed: {type(error).__name__} (message withheld so it cannot echo the credential)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
