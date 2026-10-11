"""Rebuild a corpus manifest from the wavs already on disk.

Written after a corpus build was killed at 725 of 3 326 utterances: the wavs survived, the manifest did not
(it was written only at the end), and the cache build reads the manifest -- so a fully generated half of the
corpus was unusable until it was regenerated.

The generator is deterministic in its voice choice (`voice = voice_list[i % len(voice_list)]`) and writes
`{teacher}_{index:07d}.wav`, so the records can be reconstructed exactly from the prompt list and the voice
list -- not guessed:

    python scripts/rebuild_manifest.py --corpus data/kokoro_v3 --texts data/prompts/prompts_v3.txt --teacher kokoro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parakeet.data.teacher import TEACHERS, sha1_of_array  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild manifest.jsonl from wavs + prompts")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--teacher", default="kokoro")
    ap.add_argument("--voices", default="", help="comma-separated; default is the teacher's list")
    args = ap.parse_args()

    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_teacher_corpus import DEFAULT_VOICES  # the voice lists live with the corpus builder

    corpus = Path(args.corpus)
    wav_dir = corpus / "wav"
    texts = [t for t in Path(args.texts).read_text(encoding="utf-8").splitlines() if t.strip()]
    voices = ([v.strip() for v in args.voices.split(",") if v.strip()]
              or DEFAULT_VOICES.get(args.teacher) or [""])

    existing: dict = {}
    for path in wav_dir.glob(f"{args.teacher}_*.wav"):
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        existing[index] = path
    if not existing:
        raise SystemExit(f"no {args.teacher}_*.wav under {wav_dir}")

    lines = []
    for index in sorted(existing):
        path = existing[index]
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        text = texts[index] if index < len(texts) else ""
        lines.append(json.dumps({
            "utt_id": path.stem,
            "text": text,
            "teacher": args.teacher,
            "voice": voices[index % len(voices)],
            "wav_path": str(path.relative_to(corpus)).replace("\\", "/"),
            "sample_rate": int(sr),
            "duration_s": float(wav.shape[0] / sr),
            "tags": [],
            "license": TEACHERS[args.teacher].weights_license,
            "hash": sha1_of_array(wav),
            "token_frames": None,
        }, ensure_ascii=False))

    out = corpus / "manifest.jsonl"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hours = sum(json.loads(l)["duration_s"] for l in lines) / 3600.0
    print(f"rebuilt {len(lines)} records ({hours:.2f} h) -> {out}")
    if lines and not json.loads(lines[0])["text"]:
        print("WARNING: the first record has no text -- the prompt list does not line up with the wav indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
