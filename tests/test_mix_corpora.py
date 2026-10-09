"""Joining teacher corpora: relative paths, interleaving, weights, and preserved alignment.

"The mix training of both" is the original request, and the mixer is what finally makes it expressible
against two real teachers.  The parts that matter: waveforms must resolve from the combined directory
without copying (an hour of 48 kHz audio is hundreds of megabytes), records must keep their teacher and
their `token_frames` alignment, and the mixture must be interleaved so every cache shard carries it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def _corpus(root: Path, teacher: str, n: int, aligned: bool) -> Path:
    corpus = root / teacher / "corpus"
    (corpus / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        wav = (np.sin(np.arange(2400) * 0.05) * 0.2).astype(np.float32)
        sf.write(str(corpus / "wav" / f"{teacher}_{i}.wav"), wav, 24000)
        row = {
            "utt_id": f"{teacher}_{i}",
            "text": f"a sentence for {teacher} number {i}",
            "teacher": teacher,
            "voice": "v0",
            "wav_path": f"wav/{teacher}_{i}.wav",
            "sample_rate": 24000,
            "duration_s": 0.1,
            "license": f"{teacher} licence",
        }
        if aligned:
            row["token_frames"] = [4] * len(row["text"])
        rows.append(row)
    (corpus / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (corpus / "corpus_meta.json").write_text(
        json.dumps({"teachers": {teacher: {"name": teacher, "weights_license": f"{teacher} licence"}}}),
        encoding="utf-8",
    )
    return corpus


def test_mix_keeps_teachers_alignment_and_resolves_paths(tmp_path):
    kokoro = _corpus(tmp_path, "kokoro", 3, aligned=False)
    speechify = _corpus(tmp_path, "speechify", 2, aligned=True)
    out = tmp_path / "mixed"

    result = subprocess.run(
        [sys.executable, "scripts/mix_corpora.py",
         "--corpus", str(kokoro), "--corpus", str(speechify),
         "--manifest", "train.jsonl", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    teachers = [r["teacher"] for r in rows]
    assert set(teachers) == {"kokoro", "speechify"}
    # interleaved, not concatenated: a cache shard must not be one teacher
    assert teachers[0] != teachers[1], teachers
    # every wav resolves from the combined directory without being copied
    for row in rows:
        resolved = (out / row["wav_path"]).resolve()
        assert resolved.exists(), row["wav_path"]
        assert float(resolved.stat().st_size) > 0
    assert not (out / "wav").exists(), "waveforms are referenced, not duplicated"
    # the alignment that one teacher provided survives the join
    aligned = [r for r in rows if r.get("token_frames")]
    assert len(aligned) == 2
    assert all(len(r["token_frames"]) == len(r["text"]) for r in aligned)

    meta = json.loads((out / "corpus_meta.json").read_text(encoding="utf-8"))
    assert meta["utterances_by_teacher"] == {"kokoro": 3, "speechify": 2}
    assert meta["mix"] == {"kokoro": 3.0, "speechify": 2.0}, "weights default to the share of utterances"
    assert "kokoro" in meta["teachers"] and "speechify" in meta["teachers"]
    assert meta["teachers"]["speechify"]["weights_license"] == "speechify licence", (
        "the licence must travel with the mixture"
    )


def test_mix_honours_explicit_weights_and_caps(tmp_path):
    a = _corpus(tmp_path, "kokoro", 4, aligned=False)
    b = _corpus(tmp_path, "speechify", 4, aligned=True)
    out = tmp_path / "mixed2"
    result = subprocess.run(
        [sys.executable, "scripts/mix_corpora.py",
         "--corpus", str(a), "--corpus", str(b), "--manifest", "train.jsonl",
         "--out", str(out), "--weights", "kokoro=0.7,speechify=0.3",
         "--limit-per-teacher", "2"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    meta = json.loads((out / "corpus_meta.json").read_text(encoding="utf-8"))
    assert meta["mix"] == {"kokoro": 0.7, "speechify": 0.3}
    assert meta["utterances_by_teacher"] == {"kokoro": 2, "speechify": 2}
