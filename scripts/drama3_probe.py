"""Call the Drama-3 Space and report exactly what it returns.

The teacher backend speaks the JSON {"inputs": ...} contract, which a Gradio Space cannot serve, so this probes the
Space's own `/perform` endpoint and reports the audio format, sample rate, duration and byte size rather than assuming
any of them. It is also the basis of the reliability measurement: Drama-3's renditions of the SAME text are what decide
whether it is a better teacher than Kokoro, and this Space produces a fresh rendition per call.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gradio_client import Client


def token() -> str | None:
    path = Path(".secrets/drama3.key")
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def main() -> int:
    text = " ".join(sys.argv[1:]) or "The lamp stood on the sill, and the wind had stopped."
    client = Client("fishaudio/Drama-3", token=token(), verbose=False)
    print("  connected; calling /perform with mode=Surprise")
    result = client.predict(
        text=text,
        mode="Surprise",
        voice="clone1",
        transcript="",
        name="",
        description="",
        scene="",
        temperature=0.7,
        top_p=0.7,
        top_k=30,
        api_name="/perform",
    )
    for index, item in enumerate(result):
        if isinstance(item, str) and Path(item).exists():
            size = Path(item).stat().st_size
            print(f"  return[{index}]: file {Path(item).name}  {size} bytes  suffix {Path(item).suffix}")
            try:
                import soundfile as sf

                info = sf.info(item)
                print(f"      audio: {info.samplerate} Hz, {info.channels} ch, {info.duration:.2f} s, {info.format}")
            except Exception as error:
                print(f"      not readable by soundfile: {type(error).__name__} (format may be mp3)")
        else:
            shown = str(item)[:70]
            print(f"  return[{index}]: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
