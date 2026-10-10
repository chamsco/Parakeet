"""Render a listenable demo pack from a checkpoint, with the honest comparison next to it.

    python scripts/demo_pack.py --checkpoint runs/long_v2/distill-audio_last.pt --out docs/demo

"Usable" has to mean something a person can judge by listening, and the numbers this project reports
(WER, DNSMOS, log-mel cosine) are proxies that have already misled once: round 30 measured a mel cosine
of 0.95 against the reference while the speech was unintelligible.  So the demo prints, for the same
sentences:

* the **teacher** reference (what the model is trying to match),
* the **autoencoder round trip** (latent -> waveform through the same decoder, the part measured to be
  intelligible),
* the **student** (text -> speech),

with the measured WER for each, and says plainly which are intelligible.  An index page is written next
to the audio so the files can be played in order.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: sentences the demo always renders: short, ordinary English, disjoint from nothing in particular --
#: the point is to let a listener judge, and the WER row below says whether it is intelligible
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She opened the window and listened to the rain.",
    "We should leave before the traffic gets bad.",
    "He wrote the letter twice and sent the second one.",
    "The little engine climbed the hill without stopping.",
    "Dinner is at seven, so do not be late.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a demo pack with references and proxies")
    ap.add_argument("--checkpoint", default="runs/long_v2/distill-audio_last.pt")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--corpus", default="data/speechify_corpus/corpus")
    ap.add_argument("--manifest", default="val.jsonl")
    ap.add_argument("--text-mode", default=None, choices=["char", "phoneme"])
    ap.add_argument("--out", default="docs/demo")
    ap.add_argument("--sentences", type=int, default=len(SENTENCES))
    ap.add_argument("--steps", type=int, default=None,
                    help="flow sampler NFE.  The flow variant defaults to cfg.flow.nfe (32), which runs "
                         "slower than real time; a distilled sampler uses 2-4")
    ap.add_argument("--voice", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf

    from parakeet.audio.mel import MelSpectrogram
    from parakeet.config import load_config
    from parakeet.eval.metrics import OptionalMetric, dnsmos_score, whisper_wer
    from parakeet.inference import Synthesizer
    from parakeet.models import build_model
    from parakeet.train.common import infer_model_geometry

    cfg = load_config(args.config)
    if args.text_mode:
        cfg.text.mode = args.text_mode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    geometry = infer_model_geometry(state)
    if "n_voices" in geometry:
        cfg.n_voices = geometry["n_voices"]
    if "latent_head_width" in geometry:
        cfg.autoencoder.latent_rate = max(
            1, geometry["latent_head_width"] // int(cfg.autoencoder.latent_dim)
        )
    model = build_model(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()

    texts = SENTENCES[: args.sentences]
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    mel = MelSpectrogram(cfg.audio)

    # a real teacher utterance, so the reference row is the actual target rather than an idealisation
    corpus = Path(args.corpus)
    reference_audio = None
    reference_text = None
    manifest = corpus / args.manifest
    if manifest.exists():
        from parakeet.data.text import is_prose_like

        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        # a *prose* record: a chapter heading's written form cannot match a transcript, and preferring
        # prose keeps the control meaningful
        rows = [r for r in rows if is_prose_like(r.get("text", ""))] or rows
        if rows:
            wav, rate = sf.read(str(corpus / rows[0]["wav_path"]), dtype="float32")
            reference_audio = torch.from_numpy(wav).reshape(1, -1)
            reference_text = rows[0]["text"]
            if rate != cfg.audio.sample_rate:
                target = int(reference_audio.shape[-1] * cfg.audio.sample_rate / rate)
                reference_audio = torch.nn.functional.interpolate(
                    reference_audio[:, None, :], size=target, mode="linear", align_corners=False
                )[:, 0, :]

    entries: List[Dict] = []
    student_audio: List[torch.Tensor] = []
    for index, text in enumerate(texts):
        student = synth.synthesize(text, seed=0, voice=args.voice, steps=args.steps)
        path = out / f"student_{index:02d}.wav"
        sf.write(str(path), student.detach().reshape(-1).numpy(), cfg.audio.sample_rate)
        student_audio.append(student.reshape(-1))
        # the autoencoder round trip: encode the *teacher's* own audio and decode it back.  This is the
        # acoustic path alone, which earlier rounds measured intelligible, so it separates "the decoder
        # cannot speak" from "the text side cannot drive it".
        roundtrip_path = None
        if reference_audio is not None and index == 0:
            log_mel = mel.log_mel(reference_audio)
            latent = model.autoencoder.encode(log_mel)
            back = model.autoencoder.decode(latent, length=reference_audio.shape[-1])
            roundtrip_path = out / "autoencoder_roundtrip.wav"
            sf.write(str(roundtrip_path), back.detach().reshape(-1).numpy(), cfg.audio.sample_rate)
        entries.append({"index": index, "text": text, "student": path.name,
                        "roundtrip": roundtrip_path.name if roundtrip_path else None})

    # Measured, not asserted.  `whisper_wer` assumes the control and the candidate share
    # the same texts, which is not the case here -- the control is the *teacher's own* utterance -- so
    # the two are run separately and the project's reporting rule is applied explicitly: if the
    # recogniser cannot read real speech, the candidate's number is withheld rather than reported.
    control = None
    if reference_audio is not None and reference_text:
        control = whisper_wer(
            [reference_audio.reshape(-1)], [reference_text], sample_rate=cfg.audio.sample_rate
        )
    candidate = whisper_wer(student_audio, texts, sample_rate=cfg.audio.sample_rate)
    control_ok = control is None or (control.available and float(control.value) < 0.5)
    wer = candidate if control_ok else OptionalMetric(
        None, False, f"withheld: the control on real speech failed (WER {control.value})"
    )
    mos = dnsmos_score(student_audio, sample_rate=cfg.audio.sample_rate)
    reference_wer = control

    # the autoencoder round trip gets the same treatment as the student: if it is intelligible then the
    # acoustic path works and the text side is the whole problem
    roundtrip_wer = None
    roundtrip_file = out / "autoencoder_roundtrip.wav"
    if roundtrip_file.exists() and reference_text:
        back, _ = sf.read(str(roundtrip_file), dtype="float32")
        roundtrip_wer = whisper_wer(
            [torch.from_numpy(back).reshape(-1)], [reference_text],
            sample_rate=cfg.audio.sample_rate,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "step": payload.get("step"),
        "latent_rate": cfg.autoencoder.latent_rate,
        "text_mode": cfg.text.mode,
        "sentences": len(texts),
        "student_wer": wer.value if wer.available else None,
        "student_wer_note": wer.reason if not wer.available else "control passed",
        "control_wer_on_real_speech": control.value if control and control.available else None,
        "autoencoder_roundtrip_wer": (
            roundtrip_wer.value if roundtrip_wer and roundtrip_wer.available else None
        ),
        "autoencoder_roundtrip_text": reference_text,
        "student_dnsmos": mos.value if mos.available else None,
        "reference_wer": reference_wer.value if reference_wer and reference_wer.available else None,
        "verdict": (
            "not intelligible yet: the demo says so, and shows the autoencoder round trip instead, "
            "which is the part that measurably works"
            if wer.value is None or wer.value >= 0.9
            else "intelligible on these sentences"
        ),
    }
    (out / "index.json").write_text(json.dumps({**summary, "entries": entries}, indent=2),
                                   encoding="utf-8")

    rows = "\n".join(
        f"    <tr><td>{html.escape(e['text'])}</td>"
        f"<td><audio controls src=\"{e['student']}\"></audio></td></tr>"
        for e in entries
    )
    roundtrip_row = ""
    if any(e["roundtrip"] for e in entries):
        roundtrip_row = (
            '<h2>Autoencoder round trip (the part that works)</h2>'
            '<audio controls src="autoencoder_roundtrip.wav"></audio>'
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Parakeet demo</title>
<style>body{{font-family:system-ui,sans-serif;max-width:52rem;margin:2rem auto;line-height:1.5}}
td{{padding:.4rem .6rem;border-bottom:1px solid #eee}} audio{{height:2rem}}</style></head>
<body>
<h1>Parakeet: Tiny ({summary['latent_rate']} latents/token, {summary['text_mode']} input)</h1>
<p><strong>Measured:</strong> student WER {summary['student_wer']} 
(DNSMOS {summary['student_dnsmos']}), reference/teacher WER {summary['reference_wer']}.</p>
<p><strong>Verdict:</strong> {html.escape(summary['verdict'])}</p>
{roundtrip_row}
<h2>Text to speech</h2>
<table><tr><th>sentence</th><th>rendered</th></tr>
{rows}
</table>
<p>Regenerate with <code>scripts/demo_pack.py</code>.  WER comes from a control-first recogniser
run: if the control on real speech fails, the number is withheld rather than reported.</p>
</body></html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")

    print(f"  checkpoint step {summary['step']} | rate {summary['latent_rate']} | "
          f"mode {summary['text_mode']}")
    print(f"  student WER {summary['student_wer']} | DNSMOS {summary['student_dnsmos']} | "
          f"teacher WER {summary['reference_wer']}")
    print(f"  {summary['verdict']}")
    print(f"demo -> {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
