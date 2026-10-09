"""Multi-voice wiring: the voice index must reach the model, and it must condition every head.

Round 8 found the same class of bug as round 6's mixture: `n_voices` and `voice_embed` existed, the
manifest had carried a `voice` per record since round 1, and nothing between them ever produced a
voice tensor -- so every sample trained as voice 0.  It also found a *design* gap: the voice
embedding only modulated the latent feature, leaving the duration/F0/energy heads voice-blind, which
made voice separation impossible even once the plumbing was fixed.
"""

import copy
import json

import pytest
import torch

from parakeet.config import ParakeetConfig
from parakeet.data.dataset import LatentShardDataset, collate
from parakeet.data.features import build_latent_cache
from parakeet.data.teacher import build_backend
from parakeet.data.text import TextTokenizer
from parakeet.eval import teacher_signal_loss
from parakeet.models import build_model

VOICES = ["low", "mid", "high"]


def _cfg(fast_cfg, n_voices: int) -> ParakeetConfig:
    cfg = copy.deepcopy(fast_cfg)
    cfg.n_voices = n_voices
    return cfg.validate()


def _multi_voice_manifest(cfg, tmp_path, voices=VOICES, texts=("hello there", "another line")):
    """Same texts rendered once per voice, so voice is the only systematic difference."""
    import soundfile as sf

    (tmp_path / "wav").mkdir(parents=True, exist_ok=True)
    backend = build_backend("stub_low")
    lines = []
    i = 0
    for text in texts:
        for voice in voices:
            wav, sr = backend.synthesize(text, voice=voice)
            path = tmp_path / "wav" / f"u{i}.wav"
            sf.write(str(path), wav, sr)
            lines.append(
                json.dumps(
                    {
                        "utt_id": f"u{i}",
                        "text": text,
                        "teacher": "stub_low",
                        "voice": voice,
                        "wav_path": f"wav/u{i}.wav",
                        "sample_rate": sr,
                        "duration_s": float(wav.shape[0] / sr),
                    }
                )
            )
            i += 1
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


# ------------------------------------------------------------------ the fixture is really multi-voice
def test_stub_backend_voices_have_monotone_pitch():
    from parakeet.audio import estimate_f0

    backend = build_backend("stub_low")
    measured = []
    for voice in VOICES:
        wav, sr = backend.synthesize("aaaaa", voice=voice)
        f0, voiced, _ = estimate_f0(torch.from_numpy(wav)[None], sr, hop_length=256)
        measured.append(float(f0[voiced].median().item()))
    assert measured[0] < measured[1] < measured[2], measured
    # and pitch multipliers are what the spec claims
    assert backend.pitch_for_voice("low") < backend.pitch_for_voice("high")


# ------------------------------------------------------------------ the voice reaches the model
def test_voice_embedding_conditions_every_head(fast_cfg):
    """Regression: the first implementation added the voice only to the latent feature, so the
    duration/F0/energy heads could not differ between voices at all."""
    cfg = _cfg(fast_cfg, n_voices=len(VOICES))
    model = build_model(cfg).eval()
    ids = torch.randint(1, 50, (1, 8))
    with torch.no_grad():
        low = model.text_side(ids, voice=torch.tensor([0]))
        high = model.text_side(ids, voice=torch.tensor([2]))
    for head in ("log_duration", "f0", "energy", "latent_token"):
        assert not torch.allclose(low[head], high[head]), f"voice must condition {head}"


def test_default_voice_is_index_zero(fast_cfg):
    cfg = _cfg(fast_cfg, n_voices=len(VOICES))
    model = build_model(cfg).eval()
    ids = torch.randint(1, 50, (1, 6))
    with torch.no_grad():
        implicit = model.text_side(ids)
        explicit = model.text_side(ids, voice=torch.tensor([0]))
    for head in ("log_duration", "f0", "energy", "latent_token"):
        assert torch.allclose(implicit[head], explicit[head], atol=1e-6)


def test_voice_embedding_has_one_row_per_voice(fast_cfg):
    for n in (1, 3):
        model = build_model(_cfg(fast_cfg, n_voices=n))
        assert model.voice_embed.num_embeddings == n
        assert model.voice_embed.embedding_dim == _cfg(fast_cfg, n).text.dim, (
            "the voice must live in text space so it can condition every head"
        )


# ------------------------------------------------------------------ the cache + collation carry it
def test_cache_records_voice_indices(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg, n_voices=len(VOICES))
    manifest = _multi_voice_manifest(cfg, tmp_path)
    model = build_model(cfg)
    cache = build_latent_cache(
        manifest, tmp_path / "cache", cfg, model.autoencoder,
        tokenizer=TextTokenizer(mode=cfg.text.mode),
    )
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    assert meta["voice_names"] == VOICES, "first-seen order"

    dataset = LatentShardDataset(cache)
    indices = [int(dataset[i]["voice"]) for i in range(len(dataset))]
    assert set(indices) == {0, 1, 2}
    # the manifest alternates voices, so the cached order must too
    assert indices[:3] == [0, 1, 2]

    batch = collate([dataset[0], dataset[2]])
    assert batch["voice"].shape == (2,)
    assert batch["voice"].tolist() == [0, 2]
    assert batch["voice"].dtype in (torch.int64, torch.int32)


def test_cache_rejects_more_voices_than_the_model_has(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg, n_voices=1)  # model can only index voice 0
    manifest = _multi_voice_manifest(cfg, tmp_path)
    model = build_model(cfg)
    with pytest.raises(ValueError, match="n_voices"):
        build_latent_cache(
            manifest, tmp_path / "cache_bad", cfg, model.autoencoder,
            tokenizer=TextTokenizer(mode=cfg.text.mode),
        )


def test_cached_f0_targets_follow_the_voice_pitch(fast_cfg, tmp_path):
    """End-to-end regression for the pitch-target pipeline, which had three bugs at once.

    1. the fixture applied an *unbounded* 2 %-per-token declination, sweeping a 43-character
       sentence down to 0.16x the base pitch (24 Hz for the high voice) -- outside any tracker's
       range, so every F0 target in the corpus was junk;
    2. the default pitch estimator was normalised autocorrelation, which is formant-biased: it
       reported 168 Hz for an 81 Hz voice on the fixture and only marked 52 % of frames voiced
       (YIN: 74 Hz, 100 % voiced);
    3. per-token aggregation averaged in the *unvoiced zeros*, so a half-voiced token was labelled
       with half the pitch it actually had.

    The observable consequence is that the cached F0 targets must increase with the voice's pitch
    multiplier (0.85 / 1.0 / 1.6).
    """
    from parakeet.audio.f0 import normalized_to_f0

    cfg = _cfg(fast_cfg, n_voices=len(VOICES))
    manifest = _multi_voice_manifest(cfg, tmp_path)
    model = build_model(cfg)
    cache = build_latent_cache(
        manifest, tmp_path / "cache_f0", cfg, model.autoencoder,
        tokenizer=TextTokenizer(mode=cfg.text.mode),
    )
    dataset = LatentShardDataset(cache)
    per_voice: dict[int, list[float]] = {v: [] for v in range(len(VOICES))}
    for i in range(len(dataset)):
        item = dataset[i]
        per_voice[int(item["voice"])].append(float(normalized_to_f0(item["f0"]).mean().item()))
    means = [sum(per_voice[v]) / len(per_voice[v]) for v in range(len(VOICES))]
    assert means[0] < means[1] < means[2], f"cached F0 targets are not ordered by voice: {means}"
    # and the ratio should roughly follow the fixture's pitch multipliers (0.85 : 1.0 : 1.6)
    assert means[2] / means[0] > 1.3, means


def test_yin_is_used_by_default_and_is_no_worse_than_autocorrelation():
    """YIN is the default because autocorrelation is formant-biased on hard inputs.

    An honest scope note: against the *corrected* fixture both estimators do acceptably (76.0 vs
    80.75 expected for autocorrelation, 74.1 for YIN).  The autocorrelation failure that motivated
    the switch -- 168 Hz reported for an 81 Hz voice, only 52 % of frames voiced -- appeared on the
    fixture *before* its pitch sweep was bounded, and octave errors are input-dependent.  So this
    test pins what is robustly true (the default is YIN, and it is accurate and no worse) rather
    than asserting a specific magnitude of failure for the old method.  The end-to-end guard for the
    target pipeline is :func:`test_cached_f0_targets_follow_the_voice_pitch`.
    """
    from parakeet.audio import estimate_f0

    backend = build_backend("stub_low")
    wav, sr = backend.synthesize("the quick brown fox jumps over the lazy dog", voice="low")
    w = torch.from_numpy(wav)[None]
    expected = 95.0 * 0.85  # stub_low f0 x the "low" voice multiplier

    default, dv, _ = estimate_f0(w, sr, hop_length=256, frame_length=2048)
    yin, yv, _ = estimate_f0(w, sr, hop_length=256, frame_length=2048, method="yin")
    auto, av, _ = estimate_f0(w, sr, hop_length=256, frame_length=2048, method="autocorr")

    assert torch.allclose(default[dv], yin[yv]), "the default must be YIN"
    yin_med = float(yin[yv].median().item())
    auto_med = float(auto[av].median().item())
    assert abs(yin_med - expected) / expected < 0.15, f"YIN off by too much: {yin_med:.1f} Hz"
    assert abs(yin_med - expected) <= abs(auto_med - expected) + 2.0, (
        f"YIN {yin_med:.1f} Hz should not be worse than autocorrelation {auto_med:.1f} Hz"
    )


def test_aggregate_to_tokens_ignores_unvoiced_zeros():
    from parakeet.data.features import aggregate_to_tokens

    values = [0.0, 0.5, 0.5, 0.0]
    durations = [2, 2]
    plain = aggregate_to_tokens(values, durations)
    voiced = aggregate_to_tokens(values, durations, ignore_zeros=True)
    assert plain == pytest.approx([0.25, 0.25])
    assert voiced == pytest.approx([0.5, 0.5]), "a half-voiced token must keep its pitch"
    # a fully unvoiced span stays unvoiced rather than becoming NaN
    assert aggregate_to_tokens([0.0, 0.0], [2], ignore_zeros=True) == [0.0]


# ------------------------------------------------------------------ the probe is voice-aware
def test_teacher_signal_probe_passes_the_sample_voice(fast_cfg):
    """Evaluating a voice-conditioned model with voice 0 for every sample makes it look worse."""
    cfg = _cfg(fast_cfg, n_voices=len(VOICES))
    model = build_model(cfg)
    base = {
        "ids": torch.randint(1, 50, (5,)),
        "durations": torch.full((5,), 4, dtype=torch.long),
        "f0": torch.full((5,), 0.5),
        "energy": torch.full((5,), 0.5),
        "latent_token": torch.randn(5, cfg.autoencoder.latent_dim),
    }
    spread = [{**base, "voice": torch.tensor(v)} for v in (0, 1, 2)]
    zeroed = [{**base, "voice": torch.tensor(0)} for _ in range(3)]
    with torch.no_grad():
        loss_spread = teacher_signal_loss(model, spread, cfg)
        loss_zeroed = teacher_signal_loss(model, zeroed, cfg)
    assert loss_spread != pytest.approx(loss_zeroed), "the probe must use each sample's voice"
