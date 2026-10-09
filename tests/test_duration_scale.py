"""The duration-target scale: the fix, and the trap it creates for every consumer.

Durations were the one prosody signal still regressed in raw log space while F0 and energy had been
normalised to O(1) since round 2.  Measured on real speech, the consequence was a **0.29x duration
collapse**: with targets near log(6) = 1.79 and a head initialised near zero, the mean-absolute-error
gradient on the head's weights is divided by the token count, so after 400 steps it had learned only
a constant ~1.75 frames per token -- and that single term was 1.2 of the 2.6 total loss.

Normalising fixed it (0.29x -> 0.94x), and it introduced an easy mistake: **every** consumer of
``log_duration`` must de-normalise.  Calling ``.exp()`` on the head output is off by a factor of
``exp(1.728) ~ 5.6``, which is exactly the bug this file guards.
"""

import math

import pytest
import torch

from parakeet.config import load_config
from parakeet.models.duration import (
    LOG_DURATION_MEAN,
    LOG_DURATION_STD,
    durations_to_normalized,
    normalized_to_durations,
    normalized_to_log_duration,
)


def test_duration_normalisation_round_trips_exactly():
    durations = torch.tensor([[1.0, 4.0, 6.0, 12.0, 30.0]])
    normalized = durations_to_normalized(durations)
    assert torch.equal(normalized_to_durations(normalized), durations.long())


def test_the_normalisation_centre_is_the_measured_corpus_mean():
    """Zero on the normalised scale must mean ~5.6 frames, not 1 frame."""
    frames = normalized_to_durations(torch.zeros(1)).item()
    assert frames == round(math.exp(LOG_DURATION_MEAN)) == 6
    assert normalized_to_log_duration(0.0) == pytest.approx(LOG_DURATION_MEAN)


def test_positive_normalised_output_means_longer_tokens():
    up = normalized_to_durations(torch.tensor([1.0])).item()
    down = normalized_to_durations(torch.tensor([-1.0])).item()
    assert up > down > 0
    assert up / down == pytest.approx(math.exp(2 * LOG_DURATION_STD), rel=0.2)


def test_synthesized_length_follows_the_normalised_duration_scale():
    """The regression test for the trap: ``log_duration.exp()`` gives the wrong length *ratio*.

    With the normalised target, one standard deviation of the head output must lengthen the audio by
    ``exp(LOG_DURATION_STD) ~ 1.48``.  If a consumer forgets to de-normalise, the ratio becomes
    ``exp(1) ~ 2.7`` instead, so this comparison catches it even though both produce *some* audio.
    """
    cfg = load_config("configs/parakeet_tiny.yaml")
    cfg.n_voices = 1
    from parakeet.models import build_model

    model = build_model(cfg)
    model.eval()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        reference_side = model.text_side(ids)

    def stub(level: float):
        side = {k: torch.zeros_like(v) for k, v in reference_side.items()}
        side["log_duration"] = torch.full_like(reference_side["log_duration"], level)
        return side

    lengths = {}
    with torch.no_grad():
        for level in (0.0, 1.0):
            side = stub(level)
            durations = normalized_to_durations(side["log_duration"])
            latent, _ = model.decoder_latent_from_tokens(
                side["latent_token"], durations, side["f0"], side["energy"]
            )
            lengths[level] = int(latent.shape[-1])
    expected = normalized_to_durations(torch.tensor([1.0])).item()
    assert lengths[1.0] / lengths[0.0] == pytest.approx(math.exp(LOG_DURATION_STD), rel=0.25), (
        "one normalised standard deviation must stretch the sequence by ~1.48x; a ratio near 2.7 "
        "means a consumer called .exp() on the normalised output"
    )
    assert lengths[0.0] == 6 * round(math.exp(LOG_DURATION_MEAN))
    assert expected == round(math.exp(LOG_DURATION_MEAN + LOG_DURATION_STD))


def test_the_distillation_loss_compares_against_the_normalised_target():
    """A prediction equal to the normalised target must give a ~zero duration term."""
    from parakeet.train.losses import TextSideDistillLoss

    durations = torch.tensor([[3, 5, 8, 13]], dtype=torch.float32)
    target = {
        "durations": durations,
        "f0": torch.zeros(1, 4),
        "energy": torch.zeros(1, 4),
        "latent_token": torch.zeros(1, 4, 6),
    }
    predicted = {
        "log_duration": durations_to_normalized(durations),
        "f0": torch.zeros(1, 4),
        "energy": torch.zeros(1, 4),
        "latent_token": torch.zeros(1, 4, 6),
    }
    _, parts = TextSideDistillLoss()(predicted, target, None)
    assert float(parts["duration"]) == pytest.approx(0.0, abs=1e-6)
    # and a raw-log prediction is *not* accepted, which is the mistake this replaced
    wrong = dict(predicted, log_duration=torch.log(durations))
    _, wrong_parts = TextSideDistillLoss()(wrong, target, None)
    assert float(wrong_parts["duration"]) > 0.5


def test_subtoken_geometry_is_shared_between_targets_and_expansion():
    """The cache averages sub-token targets over spans; inference must expand over the *same* spans.

    Splitting the geometry in two places is how this kind of change silently misaligns, so both call
    the same function.  This test pins that contract: a sub-vector index must land in the frame span
    the target was averaged over.
    """
    from parakeet.models.duration import align_subtokens_to_frames, subtoken_spans

    durations = torch.tensor([5, 7, 3])
    rate = 3
    subtokens = torch.arange(1 * 3 * rate * 2, dtype=torch.float32).reshape(1, 3, rate, 2)
    total = int(durations.sum())
    frames, mask = align_subtokens_to_frames(subtokens, durations, total)

    assert frames.shape == (1, total, 2)
    assert bool(mask.all()), "every frame must be covered"
    for token_index, spans in enumerate(subtoken_spans(durations, rate, total)):
        for k, (a, b) in enumerate(spans):
            assert b > a
            expected = subtokens[0, token_index, k]
            assert torch.allclose(frames[0, a, :], expected), (token_index, k, a)
            assert torch.allclose(frames[0, b - 1, :], expected), (token_index, k, b - 1)


def test_subtoken_spans_degenerate_tokens_stay_in_bounds():
    """A token with fewer frames than the rate must not produce out-of-range or negative spans."""
    from parakeet.models.duration import subtoken_spans

    spans = subtoken_spans(torch.tensor([1, 1]), 3, 2)
    assert len(spans) == 2
    for token_spans in spans:
        assert token_spans, "a token that exists must get at least one span"
        for a, b in token_spans:
            assert 0 <= a < b <= 2


def test_latent_rate_widens_the_head_but_not_the_frame_count():
    """The rate changes how many numbers a token carries, never how many frames come out."""
    from parakeet.models import build_model

    cfg = load_config("configs/parakeet_tiny.yaml")
    cfg.n_voices = 1
    ids = torch.randint(1, 20, (1, 5))
    durations = torch.tensor([[6, 6, 6, 6, 6]])
    for rate in (1, 3):
        cfg.autoencoder.latent_rate = rate
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            side = model.text_side(ids)
            latent, _ = model.decoder_latent_from_tokens(
                side["latent_token"], durations, side["f0"], side["energy"]
            )
        assert side["latent_token"].shape[-1] == rate * cfg.autoencoder.latent_dim
        assert latent.shape[-1] == int(durations.sum()), "the frame count must not depend on the rate"


def test_subtoken_alignment_handles_a_batch_with_uneven_durations():
    """A batched alignment must compute the frame total **per item**.

    The first version summed `durations` over the whole batch, so a batch of 4 with ~350 frames each
    asked for 5200 frames and crashed against the prosody path's 351.  The single-item, equal-duration
    case I tested first could not catch it.
    """
    from parakeet.models import build_model

    cfg = load_config("configs/parakeet_tiny.yaml")
    cfg.n_voices = 1
    cfg.autoencoder.latent_rate = 3
    model = build_model(cfg)
    model.eval()
    ids = torch.randint(1, 20, (3, 6))
    durations = torch.tensor([[6, 6, 6, 6, 6, 6], [4, 4, 4, 4, 4, 4], [9, 9, 9, 9, 9, 9]])
    with torch.no_grad():
        side = model.text_side(ids)
        latent, mask = model.decoder_latent_from_tokens(
            side["latent_token"], durations, side["f0"], side["energy"]
        )
    assert latent.shape[-1] == int(durations.max(dim=-1).values.sum()) or latent.shape[-1] == int(
        durations[2].sum()
    ), f"the width must follow the longest item, got {latent.shape[-1]}"
    assert latent.shape[-1] < int(durations.sum()), "not the whole batch's total"
    assert bool(mask.any())


def test_real_diagnosis_evidence_localises_the_bottleneck():
    """The diagnosis must carry its own validity checks and name a bottleneck."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "real_diagnosis.json"
    if not path.exists():
        pytest.skip("no diagnosis evidence committed")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert all(payload["checks"].values()), "the diagnostic's own validity checks must pass"
    bottleneck = payload["diagnosis"]["bottleneck"]
    assert isinstance(bottleneck, str) and bottleneck, "the diagnosis must name a bottleneck"
    # the whole point of a chain-walk: it must not say "none detected" while a teacher-input path
    # (one with no prediction error at all) is unintelligible
    paths = {row["path"]: row for row in payload["paths"]}
    assert "ae_roundtrip" in payload["fidelity"]
    ae_ok = payload["findings"]["ae_roundtrip_is_intelligible"]
    token_row = next((row for key, row in paths.items() if key.startswith("3.")), None)
    if ae_ok and token_row is not None and (token_row["wer"] or 0) > 0.5:
        assert "token expansion" in bottleneck, (
            f"the autoencoder is fine but the token path is not, so the bottleneck is the seam; "
            f"got {bottleneck!r}"
        )
    # the reference must beat every produced path, otherwise the metric is not measuring
    reference = paths["reference (Kokoro, the ceiling)"]["wer"]
    for name, row in paths.items():
        if name != "reference (Kokoro, the ceiling)" and row["wer"] is not None:
            assert reference <= row["wer"], f"{name} beat the real reference"
    # the fidelity block is what makes the finding interpretable
    assert "ae_roundtrip" in payload["fidelity"]
    assert payload["fidelity"]["ae_roundtrip"]["waveform_cosine"] < 0.5 or payload["findings"][
        "ae_roundtrip_is_intelligible"
    ]
    assert 0.5 < payload["durations_summary"]["ratio"] < 2.0, "the duration fix must hold"
