"""Real speech, at last: the teacher runtime, the corpus path, and what it exposed.

For sixteen rounds every measurement here used synthetic fixtures, on the assumption that the
teachers needed a GPU and 6 GB of weights.  That assumption was wrong: Kokoro-82M is Apache-2.0 and
runs faster than real time on this CPU through sherpa-onnx (the official `kokoro` package cannot
install here -- misaki -> spacy -> blis has no wheel and no Rust toolchain).

These tests pin the parts that real speech changed:

* the pitch tracker's voicing threshold (0.25 discarded most of real speech);
* the unaligned duration fallback, which put token spans inside pauses and produced 0 Hz targets;
* the teacher runtime dispatch, which must not silently pick a backend that cannot import.
"""

import json
from pathlib import Path

import pytest
import torch

from parakeet.audio import estimate_f0
from parakeet.data.features import aggregate_to_tokens

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the voicing threshold
def test_yin_and_autocorr_have_their_own_threshold_defaults():
    """One shared threshold made one of the two paths wrong; they have separate defaults."""
    import inspect

    from parakeet.audio.f0 import estimate_f0_yin

    assert inspect.signature(estimate_f0_yin).parameters["threshold"].default == 0.50
    assert inspect.signature(estimate_f0).parameters["threshold"].default is None


def test_voiced_fraction_grows_with_the_threshold_while_f0_stays_stable():
    """The real-speech signature of a threshold that is too strict.

    On a voiced signal with jitter and a noise floor, raising the CMND threshold must admit *more
    voiced frames* without moving the pitch estimate.  If the estimate moved, the extra frames would
    be noise rather than speech -- which is what the round-17 measurement established before the
    default was changed to 0.50.
    """
    sr = 24000
    generator = torch.Generator().manual_seed(0)
    t = torch.arange(3 * sr, dtype=torch.float32) / sr
    harmonics = torch.arange(1, 40, dtype=torch.float32)
    # the noise-to-harmonic ratio RAMPS across the utterance: the signal starts periodic and ends
    # mostly noise, so the CMND minimum sweeps through the threshold range and the voiced fraction
    # must grow with it.  A uniformly clean signal saturates every threshold at 1.0 and tests nothing.
    ramp = torch.linspace(0.0, 1.0, t.numel())
    f0 = 140.0
    freqs = f0 * harmonics
    voiced = (harmonics.pow(-1)[:, None] * torch.sin(
        2 * torch.pi * freqs[:, None] * t[None] + 0.4 * torch.rand(39, generator=generator)[:, None] * 2 * torch.pi
    )).sum(0)
    voiced = voiced / voiced.abs().max() * 0.3
    noise = torch.randn(t.numel(), generator=generator) * 0.3
    wave = (1.0 - ramp) * voiced + ramp * noise

    strict, strict_voiced, _ = estimate_f0(wave[None], sr, hop_length=256, frame_length=2048, threshold=0.15)
    default, default_voiced, _ = estimate_f0(wave[None], sr, hop_length=256, frame_length=2048)
    lenient, lenient_voiced, _ = estimate_f0(wave[None], sr, hop_length=256, frame_length=2048, threshold=0.60)

    strict_fraction = float(strict_voiced.float().mean())
    default_fraction = float(default_voiced.float().mean())
    lenient_fraction = float(lenient_voiced.float().mean())
    assert strict_fraction < default_fraction < lenient_fraction, (
        f"voiced fraction must rise with the threshold "
        f"({strict_fraction:.2f} -> {default_fraction:.2f} -> {lenient_fraction:.2f})"
    )
    # and the pitch estimate must not move while frames are being admitted: the extra frames are the
    # partly-noisy ones from the same harmonic series, not a different (noise) pitch
    median_strict = float(strict[strict_voiced].median())
    median_lenient = float(lenient[lenient_voiced].median())
    assert abs(median_strict - median_lenient) / median_strict < 0.15, (
        f"pitch moved with the threshold ({median_strict:.1f} vs {median_lenient:.1f} Hz), "
        "so the extra frames are not voiced"
    )
    assert abs(median_strict - f0) / f0 < 0.20, f"the tracker should find {f0:.0f} Hz"


# ------------------------------------------------------------------ the unaligned fallback
def test_carry_nearest_never_emits_a_zero_pitch_for_a_voiced_token():
    """A token whose span fell in a pause must not be labelled 60 Hz."""
    values = [0.5, 0.5, 0.0, 0.0, 0.4]  # token 1 is entirely unvoiced
    durations = [2, 2, 1]
    plain = aggregate_to_tokens(values, durations, ignore_zeros=True)
    carried = aggregate_to_tokens(values, durations, ignore_zeros=True, carry_nearest=True)
    assert plain == [0.5, 0.0, 0.4], plain
    assert carried == [0.5, 0.5, 0.4], carried
    assert 0.0 not in carried, "a filled token must inherit a real pitch"


def test_carry_nearest_leaves_a_fully_unvoiced_signal_alone():
    """With no voiced token anywhere there is nothing to carry; zeros stay zeros."""
    out = aggregate_to_tokens([0.0, 0.0, 0.0], [2, 1], ignore_zeros=True, carry_nearest=True)
    assert out == [0.0, 0.0]


def test_energy_weighted_fallback_is_not_degenerate_and_follows_the_energy(fast_cfg):
    """The old uniform split gave pause-heavy utterances 0 Hz targets on real speech.

    Two properties matter and pull against each other: the split must *follow the energy* (tokens
    where the speech is) and must *not collapse* (pure equal-energy produced [56, 1, 1, 1, ...] on a
    mostly-silent signal, which is just a different way to misalign).  Hence the blend with uniform.
    """
    from parakeet.config import load_config
    from parakeet.data.features import extract_signals
    from parakeet.data.text import TextTokenizer

    cfg = load_config("configs/parakeet_tiny.yaml")
    sr = cfg.audio.sample_rate
    tokenizer = TextTokenizer(mode=cfg.text.mode)
    # a realistic ratio: ~25 tokens over 3 s (281 frames) is ~11 frames per token.  A 55-character
    # text over 1 s (94 frames) has under two frames per token and no split can be sane.
    text = "the quick brown fox jumps"
    ids = tokenizer.encode(text, add_special=False)
    n_tokens = int(ids.numel())
    assert 20 <= n_tokens <= 30

    def split(wav: torch.Tensor):
        sig = extract_signals(wav, cfg, ids)
        return list(sig.durations), sig.n_frames

    # (a) speech with a short pause: tokens must concentrate on the speech, not collapse
    wav = torch.zeros(1, 3 * sr)
    burst = int(1.2 * sr)
    wav[0, int(1.2 * sr) : int(1.2 * sr) + burst] = (
        torch.sin(2 * torch.pi * 160 * torch.arange(burst) / sr) * 0.3
    )
    durations, n_frames = split(wav)
    assert sum(durations) >= n_frames * 0.9, "the spans must cover the frames"
    assert min(durations) >= 1, "every token gets at least one frame"
    # The pathology to guard against is the *mass* collapse of equal-energy splitting, where most
    # tokens get a single frame (measured: [56, 1, 1, 1, ...]).  One large leading span is legitimate
    # -- the silence has to belong to some token -- so the assertion is on how many collapsed.
    collapsed = sum(1 for d in durations if d < 2)
    assert collapsed <= 0.3 * n_tokens, f"too many single-frame tokens: {durations}"

    # (b) mostly silence: still no mass collapse (the blend guarantees at worst half-uniform)
    quiet = torch.zeros(1, 3 * sr)
    head = int(0.4 * sr)
    quiet[0, :head] = torch.sin(2 * torch.pi * 180 * torch.arange(head) / sr) * 0.3
    quiet_durations, _ = split(quiet)
    assert sum(1 for d in quiet_durations if d < 2) <= 0.3 * n_tokens, (
        f"the split collapsed on a mostly-silent signal: {quiet_durations}"
    )

    # (c) and it must still follow the energy: the spans covering the loud region get more than an
    # average share of the frames
    starts = _span_starts(durations)
    loud = [(d, at) for d, at in zip(durations, starts) if int(1.2 * sr / 256) <= at]
    share = sum(d for d, _ in loud)
    assert share > 0.5 * sum(durations), (
        f"the loud 40 % of the signal should carry more than half the spans, got {share} of "
        f"{sum(durations)}"
    )


def _span_starts(durations):
    starts, at = [], 0
    for d in durations:
        starts.append(at)
        at += d
    return starts


# ------------------------------------------------------------------ runtime + evidence
def test_kokoro_backend_dispatch_prefers_an_importable_runtime():
    """`kokoro` cannot install here, so the resolver must land on sherpa-onnx, not crash."""
    from parakeet.data.teacher import BACKENDS, SherpaKokoroBackend, _kokoro_runtime

    runtime = _kokoro_runtime()
    assert runtime in (SherpaKokoroBackend, BACKENDS["kokoro"])
    assert BACKENDS["kokoro"] is runtime


def test_sherpa_backend_rejects_an_unknown_voice_with_the_available_list():
    """Silently synthesising with the wrong speaker would poison a corpus."""
    from parakeet.data.teacher import SherpaKokoroBackend

    backend = SherpaKokoroBackend.__new__(SherpaKokoroBackend)  # no model load needed
    backend.voices = SherpaKokoroBackend.EN_V0_19_VOICES
    assert backend._speaker_id("af_heart") == 3
    assert backend._speaker_id("af_bella") == 2
    with pytest.raises(ValueError, match="available"):
        backend._speaker_id("am_michael")


def test_dnsmos_wrapper_reads_the_overall_score_not_a_default(monkeypatch):
    """Regression: the first wrapper looked for `mos_ovrl` and silently returned 0.0.

    `speechmos.dnsmos.run` returns keys ``ovrl_mos``, ``sig_mos``, ``bak_mos``, ``p808_mos`` -- note
    the word order.  A defaulted lookup scored *every* sample 0.0, including real Kokoro speech, and
    only the teacher-vs-student control made that visible: a naturalness metric that rates a real
    teacher at zero is broken, not opinionated.
    """
    import sys
    import types

    fake = types.ModuleType("speechmos.dnsmos")

    def run(sample, sr, **kwargs):
        return {"ovrl_mos": 3.25, "sig_mos": 3.51, "bak_mos": 3.90, "p808_mos": 3.72}

    fake.run = run
    package = types.ModuleType("speechmos")
    package.dnsmos = fake
    monkeypatch.setitem(sys.modules, "speechmos", package)
    monkeypatch.setitem(sys.modules, "speechmos.dnsmos", fake)

    from parakeet.eval.metrics import dnsmos_score

    metric = dnsmos_score([torch.randn(24000) * 0.05], sample_rate=24000)
    assert metric.available, metric.reason
    assert metric.value == pytest.approx(3.25), "must read ovrl_mos, not fall back to 0.0"
    assert "ovrl" in metric.detail and "p808" in metric.detail, (
        "the sub-scores diagnose why a sample is low and must travel with the value"
    )


def test_real_training_evidence_records_a_real_baseline():
    """The first real-audio training evidence must include the numbers *and* the limits."""
    path = ROOT / "docs" / "evidence" / "real_training.json"
    if not path.exists():
        pytest.skip("no real-training evidence committed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["corpus"]["audio_seconds"] > 10.0
    assert payload["autoencoder"]["recon_after"] < payload["autoencoder"]["recon_before"]
    assert payload["text_side"]["loss_after"] < payload["text_side"]["loss_before"]
    assert 0.0 < payload["text_to_audio"]["log_mel_cosine"] <= 1.0
    joined = " ".join(payload["caveats"]).lower()
    assert "not a converged model" in joined
    assert "no perceptual metric" in joined or "utmos" in joined


def test_wer_is_withheld_when_the_recogniser_fails_its_own_control(monkeypatch):
    """A transient ASR failure produced teacher WER 0.926 where the same clips scored 0.000.

    Re-running reproduced 0.000 and the audio tensors were verified byte-identical, so the bad number
    came from the recogniser, not the corpus.  The control is what caught it, so the control now
    gates the measurement: fail it twice and the WER is *unavailable with the failure as the reason*
    rather than a number someone might quote.
    """
    from parakeet.eval import metrics

    calls = {"count": 0}

    def fake_wer(audio, texts, sample_rate=24000, model_size="large-v3"):
        calls["count"] += 1
        is_control = float(audio[0].reshape(-1)[0]) == 42.0
        if is_control:
            return metrics.OptionalMetric(0.93, True, detail="fake asr")  # control fails
        return metrics.OptionalMetric(0.10, True, detail="fake asr")

    monkeypatch.setattr(metrics, "whisper_wer", fake_wer)
    control = [torch.full((64,), 42.0)]
    student = [torch.zeros(64)]
    result = metrics.whisper_wer_with_control(student, ["hello"], control_audio=control, attempts=2)

    assert result["wer"].available is False, "a number from a failed control must not be reported"
    assert "control" in result["wer"].reason
    assert result["attempts"] == 2, "the control failure must be retried before giving up"
    assert calls["count"] == 4, "two attempts, each transcribing control then student"


def test_wer_is_reported_when_the_control_passes(monkeypatch):
    from parakeet.eval import metrics

    calls = {"count": 0}

    def fake_wer(audio, texts, sample_rate=24000, model_size="large-v3"):
        calls["count"] += 1
        is_control = float(audio[0].reshape(-1)[0]) == 42.0
        return metrics.OptionalMetric(0.0 if is_control else 0.25, True, detail="fake asr")

    monkeypatch.setattr(metrics, "whisper_wer", fake_wer)
    result = metrics.whisper_wer_with_control(
        [torch.zeros(64)], ["hello"], control_audio=[torch.full((64,), 42.0)], attempts=2
    )
    assert result["wer"].available is True
    assert result["wer"].value == pytest.approx(0.25)
    assert result["attempts"] == 1
    assert calls["count"] == 2


def test_dnsmos_clamps_out_of_range_audio_and_reports_it(monkeypatch):
    """The decoder does not guarantee `|sample| <= 1`, and the resampler can overshoot even after a
    first clamp (measured: `np.ndarray values must be between -1 and 1` from DNSMOS on one utterance
    of a token-rate sweep).  A metric should measure a hot signal rather than refuse it -- while still
    making the clipping visible, because a hot output is itself a finding."""
    import sys
    import types

    seen = {}

    def run(sample, sr, **kwargs):
        seen["max_abs"] = float(max(abs(sample.min()), abs(sample.max())))
        return {"ovrl_mos": 2.5, "sig_mos": 2.5, "bak_mos": 3.0, "p808_mos": 3.0}

    fake = types.ModuleType("speechmos.dnsmos")
    fake.run = run
    package = types.ModuleType("speechmos")
    package.dnsmos = fake
    monkeypatch.setitem(sys.modules, "speechmos", package)
    monkeypatch.setitem(sys.modules, "speechmos.dnsmos", fake)

    from parakeet.eval.metrics import dnsmos_score

    hot = torch.full((24000,), 3.7)  # far outside the range DNSMOS accepts
    metric = dnsmos_score([hot], sample_rate=24000)
    assert metric.available, metric.reason
    assert metric.value == pytest.approx(2.5)
    assert seen["max_abs"] <= 1.0, "the audio handed to DNSMOS must be in range"
    assert "clipped" in metric.detail, "and the clipping must be reported, not hidden"


def test_real_evaluation_evidence_reports_its_controls():
    """A naturalness number without the teacher control, and WER without its recogniser, are not
    results."""
    path = ROOT / "docs" / "evidence" / "real_evaluation.json"
    if not path.exists():
        pytest.skip("no real-evaluation evidence committed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    naturalness = payload["naturalness"]
    assert naturalness["student"] is not None and naturalness["teacher"] is not None
    assert naturalness["teacher"] > naturalness["student"], "the control must separate them"
    wer = payload["wer"]
    assert wer["recogniser"], "WER must name the recogniser that produced it"
    assert wer["teacher"] is not None and wer["teacher"] < 0.5, "the ASR must work on the teacher"
    assert payload["utmos"]["available"] is False
    assert payload["utmos"]["reason"], "an unavailable metric must say why"


def test_real_audio_evidence_records_its_provenance_and_limitations():
    """The first real-speech evidence must say what it is, and what it is not."""
    path = ROOT / "docs" / "evidence" / "real_audio.json"
    if not path.exists():
        pytest.skip("no real-audio evidence committed")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["teacher"]["name"] == "kokoro"
    assert payload["teacher"]["allows_training"] is True
    assert "apache" in payload["teacher"]["weights_license"].lower()
    assert payload["teacher"]["kind"] != "local_fixture", "this must be a real teacher"

    assert payload["corpus"]["audio_seconds"] > 5.0
    # the threshold finding is recorded, with both values
    pitch = payload["pitch_tracking"]
    assert pitch["voiced_default_threshold"] > pitch["voiced_old_threshold_0_25"]
    assert 70.0 < pitch["median_f0_range_hz"][0] < pitch["median_f0_range_hz"][1] < 400.0

    assert payload["curation"]["n_total"] > 0
    assert payload["cache"]["items"] > 0
    # and it must admit the duration limitation rather than hide it
    assert "UNIFORM FALLBACK" in payload["duration_targets"]
    assert "not a quality claim" in payload["caveat"]
