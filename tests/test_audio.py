"""Audio front-end: iSTFT parity, streaming OLA, mel, F0."""

import torch

from parakeet.audio import (
    MelSpectrogram,
    OLAISTFT,
    StreamingOLA,
    bins_to_f0,
    estimate_f0,
    f0_to_bins,
    frame_energy_db,
    mel_filterbank,
)


def test_istft_matches_torch_istft(tiny_cfg):
    a = tiny_cfg.audio
    wav = torch.randn(2, 12000)
    mel = MelSpectrogram(a)
    spec = mel.stft(wav)
    istft = OLAISTFT(a.n_fft, a.hop_length, a.win_length, center=True)
    rec = istft(spec)
    ref = torch.istft(
        spec, a.n_fft, a.hop_length, a.win_length, window=istft.window, center=True, length=rec.shape[-1]
    )
    assert rec.shape[-1] == ref.shape[-1]
    assert torch.allclose(rec, ref, atol=1e-4)


def test_streaming_ola_matches_offline(tiny_cfg):
    a = tiny_cfg.audio
    wav = torch.randn(1, 8000)
    spec = MelSpectrogram(a).stft(wav)
    # centre=False so both paths cover exactly tc * hop samples and are directly comparable
    offline = OLAISTFT(a.n_fft, a.hop_length, a.win_length, center=False)(spec)

    st = StreamingOLA(a.n_fft, a.hop_length, a.win_length, center=False)
    parts, i = [], 0
    while i < spec.shape[-1]:
        parts.append(st.push(spec[:, :, i : i + 5]))
        i += 5
    parts.append(st.finalize())
    streamed = torch.cat(parts, dim=-1)
    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-5)


def test_mel_filterbank_shape_and_normalisation():
    fb = mel_filterbank(24000, 1024, 80)
    assert fb.shape == (80, 513)
    assert (fb >= 0).all()
    # every filter must cover at least one FFT bin
    assert (fb.sum(dim=-1) > 0).all()


def test_log_mel_shape(tiny_cfg):
    mel = MelSpectrogram(tiny_cfg.audio)
    wav = torch.randn(1, 24000)
    out = mel.log_mel(wav)
    assert out.shape[0] == 1
    assert out.shape[1] == tiny_cfg.audio.n_mels
    assert out.shape[-1] == mel.t_frames(24000)
    assert torch.isfinite(out).all()


def test_f0_estimation_on_synthetic_tone():
    sr = 24000
    t = torch.arange(sr, dtype=torch.float32) / sr
    wav = torch.sin(2 * torch.pi * 200.0 * t)[None]
    f0, voiced, conf = estimate_f0(wav, sr, hop_length=256, frame_length=1024)
    assert voiced.float().mean() > 0.8
    est = f0[voiced].median().item()
    assert abs(est - 200.0) < 6.0, est
    assert (conf[voiced] > 0.3).all()


def test_f0_bins_roundtrip():
    f0 = torch.tensor([[0.0, 80.0, 120.0, 220.0, 400.0]])
    voiced = f0 > 0
    bins = f0_to_bins(f0, voiced)
    assert bins[0, 0].item() == 0
    back = bins_to_f0(bins)
    assert torch.allclose(back[voiced], f0[voiced], rtol=0.05, atol=3.0)


def test_frame_energy_db():
    silence = torch.zeros(1, 4800)
    loud = torch.ones(1, 4800) * 0.5
    e_sil = frame_energy_db(silence).mean()
    e_loud = frame_energy_db(loud).mean()
    assert e_loud > e_sil + 50
