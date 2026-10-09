"""Audio curation pipeline: objective measurements + the published filtering thresholds."""

import copy
import json

import pytest
import torch

from parakeet.data.curate import (
    CurateConfig,
    analyze_audio,
    apply_gates,
    apply_punctuation_gaps,
    average_pairwise_wer,
    curate_manifest,
    estimate_snr_db,
    length_ratio_mask,
    normalize_loudness,
    spectral_stats,
    word_error_rate,
)
from parakeet.data.synthetic import make_corpus

SR = 24000


def _speechlike(seconds: float = 4.0, amp: float = 0.3, noise: float = 1e-4, seed: int = 0) -> torch.Tensor:
    """Bursty, harmonically rich signal.

    Deliberately *not* a pure tone: it needs bandwidth (to pass the narrowband gate) and an
    observable noise floor in pauses (so the percentile SNR estimate is measurable at all).
    ``amp/noise`` sets the signal-to-noise ratio directly, because the noise is added after peak
    normalisation.
    """
    n = int(SR * seconds)
    t = torch.arange(n, dtype=torch.float32) / SR
    sig = torch.zeros(n)
    for f in (500.0, 900.0, 1500.0, 2600.0, 4200.0, 6800.0):
        sig = sig + torch.sin(2 * torch.pi * f * t + f / 1000.0) / 6.0
    sig = sig * ((t % 0.5) < 0.40).float()  # 400 ms bursts, 100 ms pauses
    sig = sig / sig.abs().max().clamp_min(1e-6) * amp
    if noise > 0:
        sig = sig + noise * torch.randn(n, generator=torch.Generator().manual_seed(seed))
    return sig


# ------------------------------------------------------------------ measurements
def test_clean_audio_passes_every_gate():
    q = analyze_audio(_speechlike(), SR)
    assert q.keep, q.reasons
    assert 3.9 < q.duration_s < 4.1
    assert q.clipping_ratio == 0.0
    assert q.reasons == []
    assert q.snr_db is not None and q.snr_db > 40
    assert q.bandwidth99_hz > 5000


def test_short_audio_is_rejected():
    q = analyze_audio(_speechlike(0.5), SR)
    assert not q.keep
    assert any(r.startswith("too_short") for r in q.reasons)


def test_long_audio_is_rejected():
    q = analyze_audio(_speechlike(35.0), SR)
    assert not q.keep
    assert any(r.startswith("too_long") for r in q.reasons)


def test_clipped_audio_is_rejected():
    wav = _speechlike()
    wav[::10] = 1.5  # heavy clipping
    q = analyze_audio(wav, SR)
    assert not q.keep
    assert any(r.startswith("clipping") for r in q.reasons)


def test_silence_is_rejected():
    wav = _speechlike(4.0)
    wav[int(0.2 * SR) :] = 0.0  # 95 % silence
    q = analyze_audio(wav, SR)
    assert not q.keep
    assert any(r.startswith("mostly_silent") for r in q.reasons)


def test_noisy_audio_low_snr_is_rejected():
    clean = _speechlike(4.0, amp=0.3, noise=1e-4)
    # amp/noise chosen so the noise floor *is* observable (>6 dB dynamic range) while the SNR
    # lands below the 15 dB gate: with a louder noise floor the estimate correctly becomes
    # "unevaluable" instead, which is covered by the next test.
    noisy = _speechlike(4.0, amp=0.08, noise=0.012)
    snr_clean = estimate_snr_db(clean, SR)
    snr_noisy = estimate_snr_db(noisy, SR)
    assert snr_clean is not None and snr_clean > 40
    assert snr_noisy is not None and 6.0 < snr_noisy < 15.0
    assert snr_clean > snr_noisy
    q = analyze_audio(noisy, SR)
    assert not q.keep and any(r.startswith("low_snr") for r in q.reasons)
    assert analyze_audio(clean, SR).keep


def test_snr_is_reported_as_unevaluable_without_a_noise_floor():
    """Continuous speech with no pauses gives no noise reference -- the gate must be skipped."""
    t = torch.arange(SR * 4, dtype=torch.float32) / SR
    continuous = 0.3 * torch.sin(2 * torch.pi * 200 * t)
    q = analyze_audio(continuous, SR)
    assert q.snr_db is None
    assert "snr_unevaluable(no_noise_floor_observed)" in q.notes
    assert all(not r.startswith("low_snr") for r in q.reasons)


def test_all_reasons_reported_not_just_the_first():
    wav = torch.zeros(int(SR * 0.2))  # too short AND silent
    wav[::7] = 2.0  # and clipped
    q = analyze_audio(wav, SR)
    assert not q.keep
    assert len(q.reasons) >= 2


def test_mos_gate_uses_injected_predictor_only():
    cfg = CurateConfig()
    q = apply_gates(analyze_audio(_speechlike(), SR, mos_fn=lambda w, s: 4.2), cfg)
    assert q.keep and q.mos_source == "injected"

    q_bad = apply_gates(analyze_audio(_speechlike(), SR, mos_fn=lambda w, s: 2.0), cfg)
    assert not q_bad.keep
    assert any(r.startswith("low_mos") for r in q_bad.reasons)

    q_none = analyze_audio(_speechlike(), SR)
    assert q_none.mos is None and q_none.mos_source == "unavailable"


def test_spectral_stats_detect_narrowband():
    t = torch.arange(SR, dtype=torch.float32) / SR
    wide = torch.sin(2 * torch.pi * 3000 * t) + torch.sin(2 * torch.pi * 6000 * t)
    rolloff_wide, bw_wide = spectral_stats(wide, SR)
    narrow = torch.sin(2 * torch.pi * 200 * t)
    rolloff_narrow, bw_narrow = spectral_stats(narrow, SR)
    assert rolloff_wide > rolloff_narrow
    assert bw_wide > bw_narrow


# ------------------------------------------------------------------ text stages
def test_word_error_rate_bounds():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello there") == 0.5
    assert word_error_rate("hello", "") == 1.0
    assert word_error_rate("", "") == 0.0


def test_average_pairwise_wer_agreement():
    assert average_pairwise_wer(["the cat sat", "the cat sat", "the cat sat"]) == 0.0
    mixed = average_pairwise_wer(["the cat sat", "the cat sat", "a dog ran"])
    assert 0.0 < mixed <= 1.0
    assert average_pairwise_wer(["only one"]) == 0.0


def test_punctuation_gaps_add_and_remove():
    words = ["hello", "world", "again"]
    # 500 ms gap after "hello" -> comma; 10 ms gap after "world" -> no comma
    boundaries = [(0.0, 0.5), (1.0, 1.4), (1.41, 1.9)]
    text = apply_punctuation_gaps(words, boundaries)
    assert text == "hello, world again"

    words2 = ["hello,", "world"]
    boundaries2 = [(0.0, 0.5), (0.51, 1.0)]  # 10 ms gap -> existing comma removed
    assert apply_punctuation_gaps(words2, boundaries2) == "hello world"

    with pytest.raises(ValueError):
        apply_punctuation_gaps(["a"], [(0.0, 1.0), (1.0, 2.0)])


def test_length_ratio_mask_trims_tails():
    ratios = [0.01] + [1.0] * 100 + [100.0]
    mask = length_ratio_mask(ratios, low_pct=1.0, high_pct=5.0)
    assert len(mask) == len(ratios)
    assert not mask[0], "extreme low ratio must be dropped"
    assert not mask[-1], "extreme high ratio must be dropped"
    assert sum(mask) > 90
    assert length_ratio_mask([]) == []


def test_normalize_loudness_matches_cosyvoice_rule():
    wav = 0.8 * torch.ones(1000)
    out = normalize_loudness(wav, peak_factor=0.6)
    assert abs(float(out.abs().max()) - 0.6) < 1e-5
    assert torch.allclose(normalize_loudness(torch.zeros(10)), torch.zeros(10))


# ------------------------------------------------------------------ manifest driver
def test_curate_manifest_writes_kept_and_rejected(tmp_path):
    import soundfile as sf

    records = []
    for i in range(4):
        wav = _speechlike(4.0, seed=i)
        if i == 1:  # make one item too short
            wav = wav[: int(SR * 0.5)]
        path = tmp_path / f"u{i}.wav"
        sf.write(str(path), wav.numpy(), SR)
        records.append({"utt_id": f"u{i}", "wav_path": str(path), "text": "abc"})

    def load_wav(path):
        data, sr = sf.read(path, dtype="float32")
        return torch.from_numpy(data), sr

    report = curate_manifest(records, load_wav, tmp_path / "curated", asr_fn=lambda w, s: ["abc", "abc"])
    assert report.n_total == 4
    assert report.n_kept + report.n_rejected == 4, "no item may vanish without a record"
    assert report.n_kept == 3 and report.n_rejected == 1
    assert report.hours_kept > 0

    kept = [json.loads(l) for l in (tmp_path / "curated" / "kept.jsonl").read_text().splitlines() if l]
    rejected = [json.loads(l) for l in (tmp_path / "curated" / "rejected.jsonl").read_text().splitlines() if l]
    assert len(kept) == report.n_kept and len(rejected) == report.n_rejected
    assert all(r["quality"]["reasons"] for r in rejected)
    assert all("quality" in r for r in kept)
    assert (tmp_path / "curated" / "curation_report.json").exists()


def test_curate_manifest_flags_asr_disagreement(tmp_path):
    import soundfile as sf

    records = []
    for i in range(2):
        path = tmp_path / f"a{i}.wav"
        sf.write(str(path), _speechlike(4.0, seed=10 + i).numpy(), SR)
        records.append({"utt_id": f"a{i}", "wav_path": str(path)})

    def load_wav(path):
        data, sr = sf.read(path, dtype="float32")
        return torch.from_numpy(data), sr

    # wildly disagreeing ASR systems must reject the item, and be reported as such
    report = curate_manifest(
        records, load_wav, tmp_path / "c2", asr_fn=lambda w, s: ["the cat sat on the mat", "zzz"]
    )
    assert report.n_rejected == 2
    assert report.reason_counts.get("asr_disagreement", 0) == 2

    # and without an ASR stage the items are kept, with the skip recorded
    report2 = curate_manifest(records, load_wav, tmp_path / "c3")
    assert report2.n_kept == 2
    kept = [json.loads(l) for l in (tmp_path / "c3" / "kept.jsonl").read_text().splitlines() if l]
    assert kept[0]["quality"]["asr_pairwise_wer"] is None
    assert any("asr_stage_skipped" in n for n in kept[0]["quality"]["notes"])
