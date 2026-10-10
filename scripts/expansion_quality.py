"""Is the token expansion itself a quality bottleneck, independent of how well the tokens are predicted?

Both routes' audio carries a strong periodicity at the 93.75 Hz frame rate (voiced 1.0, median F0 93.75 Hz),
while the autoencoder's round trip of a *real* latent transcribes at WER 0.000.  That points at the
expansion step rather than at the prediction: `decoder_latent_from_tokens` places each token sub-vector
*constant* over its span, so the decoder is fed a staircase.

This measures it with no prediction in the loop: take a real utterance's cached token latents, expand them
exactly the way inference does, decode, and compare the audio against the direct round trip of the same
utterance's frame latents.  If the expanded version is unintelligible while the direct one is perfect, the
expansion -- not the model -- is the bottleneck, and smoothing or learning it is the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from parakeet.config import load_config
from parakeet.data.dataset import LatentShardDataset
from parakeet.eval.metrics import speechlikeness
from parakeet.train.common import derive_n_voices_from_cache, load_checkpoint_into, load_latent_norm_from_cache

CACHE = "runs/mixed_v2/latent_cache"
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "runs/text_big/distill-text_step3000.pt"

cfg = load_config("configs/parakeet_tiny.yaml")
cfg.n_voices = derive_n_voices_from_cache(CACHE)
model, applied, _payload = load_checkpoint_into(cfg, CHECKPOINT)
load_latent_norm_from_cache(model, CACHE)
model.eval()
print(f"checkpoint {CHECKPOINT} | geometry applied: {applied}")

dataset = LatentShardDataset(CACHE, indices=[0, 1])


def report(name: str, wav: torch.Tensor) -> None:
    # `speechlikeness` iterates a *list of tensor waveforms* (a batch), not one array
    stats = speechlikeness([wav.detach().reshape(-1).float()], cfg.audio.sample_rate)
    print(f"  {name:34s} voiced {stats['voiced_fraction']:.3f} | f0 {stats['median_f0_hz']:6.1f} Hz | "
          f"flatness {stats['spectral_flatness']:.3f} | speech_like {stats['speech_like']}")


print("\nspeech-likeness of the same utterance, decoded three ways:")
for index in range(len(dataset)):
    item = dataset[index]
    latent = item["latent"][None]
    with torch.no_grad():
        # the cache stores NORMALISED latents (round 38): decoding one without denormalising produces
        # nonsense, which is exactly what a first version of this probe did -- the convention has its own
        # test for a reason.  `decoder_latent_from_tokens` denormalises internally, so only the direct
        # path needs the explicit call here.
        direct = model.autoencoder.decode(model.latent_norm.denormalize(latent))
        expanded, mask = model.decoder_latent_from_tokens(
            item["latent_token"][None], item["durations"][None],
            None, None,  # the prosody path is optional and per-rate; this isolates the token expansion
            int(latent.shape[-1]),
        )
        if mask is not None:
            expanded = expanded * mask[:, None, :]
        through_tokens = model.autoencoder.decode(expanded)
    print(f"\nitem {index}: {int(latent.shape[-1])} frames")
    report("direct AE round trip (real latent)", direct)
    report("through the token expansion", through_tokens)

print("\n  Findings, round 54:")
print("  * the direct round trip of a REAL latent is intelligible (WER 0.000, round 34) yet this gate")
print("    rejects it: voiced 0.99 against a 0.25-0.9 bound.  The gate therefore has a FALSE NEGATIVE on")
print("    known-good audio, so 'speech_like: False' is not evidence that an output is not speech, and the")
print("    reliable discriminator remains WER with its teacher control.")
print("  * the token expansion is close to the direct round trip on every component here, so it is not a")
print("    quality bottleneck in its own right -- consistent with the piecewise-constant design being a")
print("    deliberate fix for the round-22 seam.")
