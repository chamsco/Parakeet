"""Say a sentence: text on the command line, a WAV on disk.

    python scripts/say.py "The lamp stood on the sill."
    python scripts/say.py --checkpoint runs/text_v5/distill-text_step10000.pt "Any words you like."

WHAT TO EXPECT, said here rather than discovered by the listener. The autoencoder's own round trip is intelligible --
WER 0.000, measured -- but the text side is not: this checkpoint reads WER 1.0000 on held-out prompts, because its
conditioned latents track the target at rho 0.391 where the recogniser needs above 0.60. So the audio this produces is
speech-shaped and rhythmic and the words are not there yet. That is the measured state of the project, not a fault in
your machine or in this script.

The model is loaded with `load_checkpoint_into`, the same helper `real_eval.py` uses, because an earlier round lost time
to an ad-hoc loader that covered only two of the four widths a checkpoint carries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.inference import Synthesizer, write_wav  # noqa: E402
from parakeet.train.common import load_checkpoint_into, load_latent_norm_from_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesise a sentence with a trained text-side checkpoint")
    ap.add_argument("text", nargs="+", help="the sentence, quoted")
    ap.add_argument("--checkpoint", default="runs/text_v5/distill-text_step10000.pt")
    ap.add_argument("--cache", default="runs/holdout_cache", help="cache whose latent normaliser the model expects")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/say.wav")
    args = ap.parse_args()

    sentence = " ".join(args.text)
    if not Path(args.checkpoint).exists():
        print(f"no checkpoint at {args.checkpoint}")
        return 2

    cfg = load_config(args.config)
    model, applied, payload = load_checkpoint_into(cfg, args.checkpoint)
    model.eval()
    if applied:
        print(f"  checkpoint geometry: {applied}")
    if args.cache and load_latent_norm_from_cache(model, args.cache):
        print(f"  latent normaliser loaded from {args.cache}")
    print(f"  checkpoint step {payload.get('step')} | n_voices {cfg.n_voices} "
          f"| latent_rate {cfg.autoencoder.latent_rate}")

    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    wav = synth.synthesize(sentence, seed=args.seed, steps=args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(str(out), wav, cfg.audio.sample_rate)
    seconds = len(wav.reshape(-1)) / cfg.audio.sample_rate
    print(f"  wrote {out}  ({seconds:.2f} s at {cfg.audio.sample_rate} Hz)")
    print("  reminder: rho 0.391 against a 0.60 threshold, so the rhythm is there and the words are not yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
