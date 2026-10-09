"""ONNX export / int8 quantisation parity tests.

Skipped automatically when ``onnx`` / ``onnxruntime`` are not installed, so the core suite stays
dependency-light.
"""

import copy

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from parakeet.audio.mel import MelSpectrogram
from parakeet.config import ParakeetConfig
from parakeet.inference.onnx_export import (  # noqa: E402
    OnnxVocoder,
    benchmark,
    compare_fp32_int8,
    export_onnx,
    quantize_int8,
)
from parakeet.models import build_model  # noqa: E402


def _cfg(fast_cfg) -> ParakeetConfig:
    cfg = copy.deepcopy(fast_cfg)
    cfg.audio.n_fft = 512
    cfg.audio.hop_length = 128
    cfg.audio.win_length = 512
    return cfg.validate()


def _latents(model, cfg, n: int = 2):
    """Encode a synthetic mel *batch* (iterating a 3-D tensor would yield 2-D slices)."""
    mels = torch.randn(n, cfg.audio.n_mels, 40) * 0.5
    with torch.no_grad():
        latent = model.autoencoder.encode(mels)
    return [latent[i : i + 1].numpy().astype(np.float32) for i in range(latent.shape[0])]


def test_onnx_decoder_matches_pytorch(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    path = export_onnx(model, cfg.audio, tmp_path / "dec.onnx", opset=17)
    assert path.exists() and path.stat().st_size > 0

    latents = _latents(model, cfg)
    voc = OnnxVocoder(path, cfg.audio)
    with torch.no_grad():
        spec_ref = model.autoencoder.spectrogram(torch.from_numpy(latents[0]))
    spec_onnx = voc.spectrogram(latents[0])
    diff = (spec_ref - spec_onnx).abs().max().item()
    assert diff < 1e-4, f"ONNX decoder drifted from PyTorch by {diff:.3e}"

    wav = voc.decode(latents[0])
    assert wav.dim() == 2 and wav.shape[-1] > 0
    assert torch.isfinite(wav).all()


def test_onnx_dynamic_time_axis(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    path = export_onnx(model, cfg.audio, tmp_path / "dyn.onnx")
    voc = OnnxVocoder(path, cfg.audio)
    for frames in (7, 23, 41):
        latent = np.random.randn(1, cfg.autoencoder.latent_dim, frames).astype(np.float32)
        wav = voc.decode(latent)
        expected = max(0, (frames - 2)) * cfg.audio.hop_length
        assert abs(wav.shape[-1] - expected) <= cfg.audio.hop_length


def test_int8_is_smaller_and_still_runs(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    latents = _latents(model, cfg, n=3)
    fp32 = export_onnx(model, cfg.audio, tmp_path / "f.onnx")
    int8 = quantize_int8(fp32, tmp_path / "i.onnx", latents)

    assert int8.stat().st_size < fp32.stat().st_size, "int8 must be smaller"
    fp32_stats = benchmark(fp32, cfg.audio, latents, runs=2)
    int8_stats = benchmark(int8, cfg.audio, latents, runs=2)
    for stats in (fp32_stats, int8_stats):
        assert stats["mean_ms"] > 0
        assert stats["model_mb"] > 0
    assert "CPUExecutionProvider" in int8_stats["providers"]


def test_compare_fp32_int8_reports_deviation(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    mel = MelSpectrogram(cfg.audio)
    calibration = [mel.log_mel(torch.randn(1, 8192) * 0.2) for _ in range(3)]
    report = compare_fp32_int8(model, cfg.audio, tmp_path / "out", calibration, runs=2)
    assert report["size_reduction_x"] > 1.0
    assert report["fp32"]["mean_ms"] > 0 and report["int8"]["mean_ms"] > 0
    # int8 must not destroy the decoder output; this bound is loose on purpose (it is a *smoke*
    # bound on an untrained model, not a quality claim)
    assert report["int8_max_log_mag_deviation"] < 2.0
    assert np.isfinite(report["int8_max_phase_deviation_rad"])


# ---------------------------------------------------------------------------------------
# text side: this is where the legacy exporter silently produces a single-length graph
# ---------------------------------------------------------------------------------------
def test_text_side_onnx_matches_pytorch_across_token_lengths(fast_cfg, tmp_path):
    pytest.importorskip("onnxscript", reason="dynamic-token export needs the dynamo exporter")
    from parakeet.inference.onnx_export import OnnxTextSide, _TextSide, export_text_side_onnx

    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    path = export_text_side_onnx(model, tmp_path / "text.onnx", opset=18, tokens=8, dynamo=True)
    side = OnnxTextSide(path)
    names = ("log_duration", "latent_token", "f0", "energy")
    with torch.no_grad():
        for length in (5, 8, 17):
            ids = torch.randint(1, cfg.text.vocab_size, (1, length))
            ref = _TextSide(model).eval()(ids)
            got = side(ids.numpy())
            for name, expected in zip(names, ref):
                dev = float(np.abs(np.asarray(got[name]) - expected.numpy()).max())
                assert dev < 1e-4, f"{name} deviates by {dev:.2e} at token length {length}"


def test_text_side_onnx_rejects_wrong_dtype(fast_cfg, tmp_path):
    """ORT silently casts a float input to int64 indices; our wrapper must not allow that."""
    pytest.importorskip("onnxscript")
    from parakeet.inference.onnx_export import OnnxTextSide, export_text_side_onnx

    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    path = export_text_side_onnx(model, tmp_path / "text2.onnx", opset=18, dynamo=True)
    side = OnnxTextSide(path)
    with pytest.raises(TypeError, match="integers"):
        side(np.zeros((1, 4), dtype=np.float32))


def test_vocoder_rejects_non_float_latent(fast_cfg, tmp_path):
    cfg = _cfg(fast_cfg)
    model = build_model(cfg).eval()
    path = export_onnx(model, cfg.audio, tmp_path / "v.onnx")
    voc = OnnxVocoder(path, cfg.audio)
    with pytest.raises(TypeError, match="floating"):
        voc.spectrogram(np.zeros((1, cfg.autoencoder.latent_dim, 5), dtype=np.int64))


def test_full_onnx_pipeline_matches_torch(fast_cfg, tmp_path):
    pytest.importorskip("onnxscript")
    from parakeet.inference.onnx_export import compare_pipelines

    cfg = _cfg(fast_cfg)
    # seed: with *random* weights the int8 deviation varies enough to make the cosine bound flaky
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    texts = ["hello world", "a slightly longer test sentence"]
    report = compare_pipelines(model, cfg, tmp_path / "pipe", texts, runs=1, threads=1)

    assert report["total_int8_mb"] < report["total_fp32_mb"], "int8 must be smaller in total"
    assert report["int8_vs_torch_waveform_cosine"] > 0.90, (
        f"int8 pipeline must track PyTorch, got cosine "
        f"{report['int8_vs_torch_waveform_cosine']:.4f} (mel L1 {report['int8_vs_torch_mel_l1']:.4f})"
    )
    assert report["int8_vs_torch_mel_l1"] < 1.0
    assert report["mean_ms"]["torch"] > 0 and report["mean_ms"]["onnx_int8"] > 0
    assert set(report["mean_ms"]) == {"torch", "onnx_fp32", "onnx_int8", "onnx_int8_shipped"}


def test_onnx_artifact_bytes_counts_external_data(tmp_path):
    from parakeet.inference.onnx_export import onnx_artifact_bytes

    graph = tmp_path / "m.onnx"
    graph.write_bytes(b"0" * 1000)
    assert onnx_artifact_bytes(graph) == 1000
    sidecar = tmp_path / "m.onnx.data"
    sidecar.write_bytes(b"0" * 500)
    assert onnx_artifact_bytes(graph) == 1500, "external weight sidecar must be counted"
