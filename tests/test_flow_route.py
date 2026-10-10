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


def test_the_velocity_field_actually_uses_its_text_conditioning():
    """An *untrained* estimator must still be sensitive to which text it is given.

    Round 39 measured the opposite: velocity correlation 0.9968 between two different texts, with the
    cross-attention branch contributing a tenth of the conv branch's strength (0.063 against 0.604), i.e.
    ~90 % of the velocity was text-independent from initialisation.  Random weights propagate their
    inputs, so an untrained network ignoring its conditioning is a wiring signal, not a training one.
    With the pooled conditioning token and a cross-attention gain this drops to ~0.94 (34 % of the
    velocity text-dependent), and this test fails if it regresses.
    """
    from parakeet.data.text import TextTokenizer
    from parakeet.models.flow import consistency_sample  # noqa: F401  (import kept for clarity)

    cfg = load_config("configs/parakeet_flow.yaml")
    cfg.n_voices = 2
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    tokenizer = TextTokenizer(mode=cfg.text.mode)

    velocities = []
    with torch.no_grad():
        for text in ("The quick brown fox jumps over the lazy dog.",
                     "Dinner is at seven, so do not be late."):
            ids, mask = tokenizer.batch([text], add_special=False)
            memory, memory_mask, _cond = model.conditions(
                ids, mask, voice=torch.zeros(1, dtype=torch.long)
            )
            generator = torch.Generator().manual_seed(0)
            x_t = torch.randn(1, cfg.flow.latent_dim * cfg.flow.compress, 60, generator=generator)
            t = torch.full((1,), 0.5)
            velocities.append(model.vf(x_t, t, memory, memory_mask))

    a, b = velocities[0].reshape(-1), velocities[1].reshape(-1)
    correlation = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    text_dependent_share = float((a - b).std() / a.std())
    assert correlation < 0.99, (
        f"an untrained estimator is nearly indifferent to its text (rho {correlation:.4f}); the "
        "conditioning path is too weak to train through"
    )
    assert text_dependent_share > 0.15, (
        f"only {text_dependent_share:.3f} of the velocity depends on the text"
    )


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
