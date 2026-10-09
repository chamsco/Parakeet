"""Inference: synthesis, streaming decoder parity, phase-lock filter, int8."""

import copy

import numpy as np
import pytest
import torch

from parakeet.audio.mel import MelSpectrogram
from parakeet.config import load_config
from parakeet.inference import (
    StreamingVocoder,
    Synthesizer,
    checkpoint_bytes,
    phase_coherence,
    phase_lock,
    quantize_weights_,
    save_int8_state_dict,
    size_report,
    write_wav,
)
from parakeet.models import build_model


@pytest.fixture()
def fast_tiny(fast_cfg):
    cfg = copy.deepcopy(fast_cfg)
    return cfg


def test_tiny_synthesize_produces_finite_audio(fast_tiny):
    model = build_model(fast_tiny)
    synth = Synthesizer(model, fast_tiny, device="cpu", apply_phase_lock=False)
    wav = synth.synthesize("hello parakeet", seed=0)
    assert wav.dim() == 2 and wav.shape[-1] > 0
    assert torch.isfinite(wav).all()


def test_phase_lock_preserves_spectral_envelope(fast_tiny):
    """Phase-only modification cannot be exactly re-analysable.

    A modified STFT is generally *inconsistent* (it is no longer the STFT of any signal), so
    re-analysing the output changes magnitudes slightly even though only phase was edited.  What
    we can and do require is that the energy and the spectral envelope are preserved.
    """
    a = fast_tiny.audio
    g = torch.Generator().manual_seed(0)
    wav = 0.2 * torch.randn(1, 8192, generator=g)
    window = torch.hann_window(a.win_length)
    before = torch.stft(wav, a.n_fft, a.hop_length, a.win_length, window, return_complex=True).abs()
    after_wav = phase_lock(wav, sample_rate=a.sample_rate, n_fft=a.n_fft, hop_length=a.hop_length)
    assert after_wav.shape[-1] == wav.shape[-1]
    after = torch.stft(after_wav, a.n_fft, a.hop_length, a.win_length, window, return_complex=True).abs()

    rms_before = wav.pow(2).mean().sqrt()
    rms_after = after_wav.pow(2).mean().sqrt()
    assert abs(float(rms_after / rms_before) - 1.0) < 0.05, "phase lock must not change loudness"

    n = min(before.shape[-1], after.shape[-1])
    b_flat = before[..., 2 : n - 2].reshape(-1)
    a_flat = after[..., 2 : n - 2].reshape(-1)
    cos = torch.nn.functional.cosine_similarity(b_flat[None], a_flat[None]).item()
    assert cos > 0.99, f"spectral envelope drifted (cosine {cos:.4f})"


def test_phase_lock_reduces_buzz(fast_tiny):
    """A magnitude spectrum with randomised phase is the definition of 'buzz'."""
    a = fast_tiny.audio
    sr = a.sample_rate
    t = torch.arange(sr, dtype=torch.float32) / sr
    pulse = torch.zeros_like(t)
    pulse[:: int(sr / 120)] = 1.0
    wav_pulse = pulse[None]
    spec = torch.stft(
        wav_pulse, a.n_fft, a.hop_length, a.win_length, torch.hann_window(a.win_length), return_complex=True
    )
    g = torch.Generator().manual_seed(1)
    rnd_phase = torch.rand(spec.shape, generator=g) * 2 * torch.pi
    buzzy = torch.istft(
        torch.polar(spec.abs(), rnd_phase),
        a.n_fft,
        a.hop_length,
        a.win_length,
        torch.hann_window(a.win_length),
        length=wav_pulse.shape[-1],
    )
    coh_before = phase_coherence(buzzy, sample_rate=sr).item()
    fixed = phase_lock(buzzy[None] if buzzy.dim() == 1 else buzzy, sample_rate=sr)
    coh_after = phase_coherence(fixed, sample_rate=sr).item()
    assert coh_after > coh_before, f"coherence {coh_before:.4f} -> {coh_after:.4f}"


def test_phase_lock_smooth_method_runs(fast_tiny):
    a = fast_tiny.audio
    wav = 0.2 * torch.randn(1, 4096)
    out = phase_lock(
        wav, sample_rate=a.sample_rate, n_fft=a.n_fft, hop_length=a.hop_length, method="smooth", smooth_frames=9
    )
    assert out.shape[-1] == wav.shape[-1]
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        phase_lock(wav, method="nonsense")


def test_streaming_vocoder_matches_offline(fast_tiny):
    """With zero-prefill the causal decoder's streaming output must equal the offline output."""
    model = build_model(fast_tiny)
    ae = model.autoencoder
    assert ae.cfg.causal_decoder
    g = torch.Generator().manual_seed(2)
    latent = torch.randn(1, fast_tiny.autoencoder.latent_dim, 40, generator=g)
    offline = ae.decode(latent)
    voc = StreamingVocoder(ae, chunk_frames=8)
    outs = [voc.push(latent[:, :, i : i + 8]) for i in range(0, 40, 8)]
    outs.append(voc.flush())
    streamed = torch.cat(outs, dim=-1)
    assert streamed.shape[-1] == offline.shape[-1]
    diff = (offline - streamed).abs().max()
    assert diff.item() < 1e-4, f"streaming/offline mismatch {diff.item():.3e}"


def test_synthesize_stream_chunks_cover_audio(fast_tiny):
    model = build_model(fast_tiny)
    synth = Synthesizer(model, fast_tiny, device="cpu", apply_phase_lock=False)
    whole = synth.synthesize("streaming test", seed=0).reshape(-1)
    chunks = list(synth.synthesize_stream("streaming test", seed=0))
    joined = np.concatenate(chunks)
    assert joined.shape[0] == whole.shape[0]


def test_int8_quantisation_and_size_report(fast_tiny):
    model = build_model(fast_tiny)
    before = size_report(model)
    weights_before = [p.detach().clone() for p in model.parameters()]
    quantize_weights_(model, bits=8, per_channel=True)
    after = size_report(model)
    assert after["int8_mb"] < before["fp32_mb"]
    changed = any(not torch.equal(a, b) for a, b in zip(weights_before, model.parameters()))
    assert changed, "quantisation did nothing"
    assert float(before["params"]) == float(after["params"])


def test_four_bit_requires_per_channel(fast_tiny):
    model = build_model(fast_tiny)
    with pytest.raises(ValueError):
        quantize_weights_(model, bits=4, per_channel=False)


def test_save_int8_state_dict(fast_tiny, tmp_path):
    model = build_model(fast_tiny)
    path = save_int8_state_dict(model, tmp_path / "int8.pt")
    assert path.exists()
    assert checkpoint_bytes(path) > 0
    payload = torch.load(path, weights_only=False)
    assert any(v.dtype == torch.int8 for v in payload.values())


def test_write_and_read_wav(fast_tiny, tmp_path):
    import soundfile as sf

    wav = 0.3 * torch.randn(1, 4800)
    path = write_wav(tmp_path / "out.wav", wav, fast_tiny.audio.sample_rate)
    data, sr = sf.read(str(path))
    assert sr == fast_tiny.audio.sample_rate
    assert data.shape[0] == 4800


def test_synthesizer_describe(fast_tiny):
    synth = Synthesizer(build_model(fast_tiny), fast_tiny, device="cpu")
    info = synth.describe()
    assert info["params"] > 0 and info["int8_mb"] > 0


def test_shipped_tiny_config_synthesizes(tiny_yaml):
    cfg = load_config(tiny_yaml)
    cfg.flow.nfe = 1
    synth = Synthesizer(build_model(cfg), cfg, device="cpu")
    wav = synth.synthesize("a short sentence.", steps=1, seed=0)
    assert wav.shape[-1] > 0
    assert torch.isfinite(wav).all()
    assert synth.buzz_metric(wav) >= 0.0


def test_phase_lock_default_grid_is_the_measured_choice():
    """The delay-grid resolution bounds the achievable lock.

    The A/B in ``scripts/phase_lock_ab.py`` measured 64 -> 256 roughly tripling the coherence the
    filter adds to glottal-locked speech while *reducing* what it adds to white noise (i.e. less of
    the effect is the filter's own arithmetic).  Both the offline and the streaming filter must ship
    that default, or the improvement never reaches the shipped path.
    """
    import inspect

    from parakeet.inference import StreamingPhaseLock, phase_lock

    assert inspect.signature(phase_lock).parameters["n_tau"].default == 256
    assert inspect.signature(StreamingPhaseLock.__init__).parameters["n_tau"].default == 256


def test_finer_delay_grid_locks_noise_less_and_speech_more():
    """Regression for the measured finding, on one second of signal so it stays cheap.

    Note what is *not* asserted: that the filter helps glottal-locked speech more than noise.  The
    A/B measured the opposite in this statistic (noise gains more coherence than speech-like phase,
    so the speech-vs-noise gap narrows), which means within-frame phase concentration cannot
    demonstrate speech-specific locking on its own.  Only claims this test can support are asserted;
    the negative result lives in scripts/phase_lock_ab.py and docs/05-VERIFICATION.md.
    """
    from parakeet.inference import phase_coherence, phase_lock

    sr = 24000
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(sr, generator=generator) * 0.05

    def coherence(signal, n_tau=None):
        filtered = (
            signal[None] if n_tau is None else phase_lock(signal[None], sample_rate=sr, n_tau=n_tau)[0][None]
        )
        return float(phase_coherence(filtered, sample_rate=sr))

    noise_gain_fine = coherence(noise, 1024) - coherence(noise)
    noise_gain_coarse = coherence(noise, 64) - coherence(noise)
    assert noise_gain_fine < noise_gain_coarse, (
        f"a finer grid must add less coherence to noise ({noise_gain_fine:.4f} vs "
        f"{noise_gain_coarse:.4f})"
    )

    # and a partially glottal-locked signal must gain *something*: the filter is not a no-op
    f0 = 120.0
    t = torch.arange(sr, dtype=torch.float32) / sr
    harmonics = torch.arange(1, 40, dtype=torch.float32)
    freqs = f0 * harmonics
    jitter = 0.3 * torch.rand(39, generator=generator) * 2 * torch.pi
    phase = -2 * torch.pi * freqs / f0 + jitter
    wave = (harmonics.pow(-1)[:, None] * torch.sin(2 * torch.pi * freqs[:, None] * t + phase[:, None])).sum(0)
    locked = wave / wave.abs().max() * 0.3
    assert coherence(locked, 1024) > coherence(locked), "the filter must have an effect on speech"
