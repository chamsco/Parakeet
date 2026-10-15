"""Pull one Drama-3 rendition and turn its HLS stream into a normal WAV.

The Space's /perform returns an HLS playlist rather than an audio file, because Drama-3 streams in real time ("First audio
1115 ms, 3.2 s audio, 2.62x real time"). ffmpeg is on this machine, so the playlist is converted to 24 kHz mono, which is
what this project's corpus and autoencoder use.

Reports the playlist's own contents as well, since whether its segments are local files or remote URLs decides whether
this conversion works on its own or needs the Space to be reachable during decoding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gradio_client import Client


def token() -> str | None:
    path = Path(".secrets/drama3.key")
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def main() -> int:
    text = " ".join(sys.argv[1:]) or "The lamp stood on the sill, and the wind had stopped."
    client = Client("fishaudio/Drama-3", token=token(), verbose=False)
    playlist, status, _choices = client.predict(
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
    print(f"  status: {status}")
    playlist_path = Path(playlist)
    print(f"  playlist: {playlist_path}  ({playlist_path.stat().st_size} bytes)")
    body = playlist_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
    print(f"  segment entries: {len(lines)}; first: {lines[0][:80] if lines else '(none)'}")
    base = playlist_path.parent
    print(f"  sibling files: {sorted(p.name for p in base.glob('*'))[:8]}")

    out = Path("runs/drama3_probe.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(playlist_path), "-ac", "1", "-ar", "24000", str(out)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not out.exists():
        print(f"  ffmpeg failed: {completed.stderr.strip()[:200]}")
        return 1
    import soundfile as sf

    info = sf.info(str(out))
    print(f"  wrote {out}: {info.samplerate} Hz, {info.channels} ch, {info.duration:.2f} s, {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
