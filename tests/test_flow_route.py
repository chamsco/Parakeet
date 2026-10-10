"""The flow route's preconditions, which a config mistake can silently break.

Round 32 put `ParakeetFlow` on real data.  Two traps were hit on the way and both are cheap to guard:

* the flow config must carry the **same autoencoder geometry as the cache it trains on**.  The cached
  latents come from the Tiny autoencoder (decoder_dim 256); `parakeet_small.yaml` carries 384, so
  training the flow there would pair a decoder with latents from an encoder it does not share.  The
  failure is loud in `--warm-start` but silent if the flow is trained from scratch;
* `ParakeetFlow.synthesize` must exist and be reachable from `Synthesizer`, because the flow variant is
  only useful if it can actually speak.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from parakeet.config import load_config
from parakeet.inference import Synthesizer
from parakeet.models import build_model

ROOT = Path(__file__).resolve().parents[1]
TRAINED_AE = ROOT / "runs/ae_scaled/adversarial/autoencoder_last.pt"


def test_flow_config_matches_the_autoencoder_that_made_the_cache():
    """The check that would have caught the 384-vs-256 trap before a run started."""
    if not TRAINED_AE.exists():
        pytest.skip("the trained autoencoder checkpoint is not present in this workspace")
    cfg = load_config("configs/parakeet_flow.yaml")
    assert cfg.variant == "small", "the flow model is the 'small' variant; tiny has no velocity field"
    model = build_model(cfg)
    assert hasattr(model, "vf") and hasattr(model, "length_predictor")

    payload = torch.load(TRAINED_AE, map_location="cpu", weights_only=False)
    state = (payload.get("ema") or {}).get("shadow") or payload["model"]
    latest = model.state_dict()
    checked = 0
    mismatched = []
    for key, value in state.items():
        if not key.startswith("autoencoder."):
            continue
        if key not in latest:
            continue
        checked += 1
        if tuple(value.shape) != tuple(latest[key].shape):
            mismatched.append(f"{key}: checkpoint {tuple(value.shape)} vs config {tuple(latest[key].shape)}")
    assert checked > 50, f"expected the autoencoder weights in the checkpoint, found {checked} keys"
    assert not mismatched, (
        "the flow config's autoencoder does not match the checkpoint that produced the cached latents: "
        + "; ".join(mismatched[:3])
    )


def test_the_flow_model_can_synthesize_through_the_shared_wrapper():
    """A trained flow that cannot be sampled is not a model, it is a checkpoint."""
    cfg = load_config("configs/parakeet_flow.yaml")
    cfg.n_voices = 2
    model = build_model(cfg)
    assert callable(getattr(model, "synthesize", None))
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=False)
    wav = synth.synthesize("hello there", voice=0, seed=0, steps=2)
    assert wav.dim() == 2 and wav.shape[0] == 1
    assert wav.numel() > 0, "sampling produced no audio at all"
    assert bool(torch.isfinite(wav).all())


def test_the_length_head_starts_near_a_plausible_length():
    """Round 32's measured bug: the head started 6.4 away from its target in log space.

    AdamW moves a parameter by roughly the learning rate per step, so an untrained head predicting 0.35
    (about one frame) against a corpus of ~850 (log 6.75) needs ~32 000 steps at lr 2e-4 just to travel
    there.  Runs of 1 600-6 000 steps never arrived -- the duration collapse measured in round 30 -- and
    training this head alone at lr 1e-2 reaches loss 0.006 in 60 steps, so the head was never the
    problem.  The bias initialisation is what closes the gap for free.
    """
    import math

    from parakeet.models.duration import UtteranceLengthPredictor

    cfg = load_config("configs/parakeet_flow.yaml")
    predictor = UtteranceLengthPredictor(cfg.duration, 64, 32)
    memory = torch.randn(2, 10, 64)
    cond = torch.randn(2, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    with torch.no_grad():
        predicted = predictor(memory, cond, mask)
    frames = predicted.exp()
    assert bool((frames > 200).all()), (
        f"the head starts at {frames.tolist()} frames; a short run cannot travel to the target region"
    )
    assert bool((frames < 2000).all()), f"the head starts absurdly long: {frames.tolist()}"
    assert abs(float(predicted.mean()) - cfg.duration.log_length_init) < 0.5
