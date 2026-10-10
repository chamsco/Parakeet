"""The real-valued iSTFT path, which is what lets the GPU train the decoder at all.

DirectML has no complex dtype, and the autoencoder's output head is `torch.polar` + `torch.fft.irfft`.
The decoder's conv stack measured **6.1x faster** on this machine's Radeon than on the CPU (67 ms ->
11 ms for 4x900 frames), so this path is the difference between "the GPU can train the acoustic stage"
and "the GPU can train everything except the part that costs the most".

The replacement is a real matrix multiply against cosine/sine bases with irfft's normalisation folded
in.  It must be numerically the same thing, not merely plausible -- an autoencoder trained through the
complex path has to keep producing the same audio -- so the test asserts equivalence against
`torch.fft.irfft` and against `torch.istft`.
"""

from __future__ import annotations

import math

import pytest
import torch

from parakeet.audio.istft import OLAISTFT, device_supports_complex, irfft_frames


@pytest.mark.parametrize("n_fft", [256, 1024])
def test_real_inverse_dft_matches_irfft(n_fft):
    """The basis multiply must reproduce irfft exactly (it is the same arithmetic)."""
    generator = torch.Generator().manual_seed(0)
    real = torch.randn(2, n_fft // 2 + 1, 7, generator=generator)
    imag = torch.randn(2, n_fft // 2 + 1, 7, generator=generator)
    # Nyquist and DC must be real for a real-signal spectrum, as irfft assumes
    imag[:, 0, :] = 0.0
    imag[:, -1, :] = 0.0

    reference = torch.fft.irfft(torch.complex(real, imag).transpose(1, 2), n=n_fft)
    # force the real path explicitly, whatever the device supports
    import parakeet.audio.istft as module

    original = module.device_supports_complex
    try:
        module.device_supports_complex = lambda device: False
        obtained = irfft_frames(real, imag, n_fft)
    finally:
        module.device_supports_complex = original
    assert torch.allclose(obtained, reference, atol=1e-4), (obtained - reference).abs().max()


def test_the_real_path_is_used_only_where_complex_is_missing():
    assert device_supports_complex(torch.device("cpu"))
    assert not device_supports_complex(torch.device("privateuseone:0")), (
        "DirectML reports privateuseone and cannot hold ComplexFloat"
    )


def test_full_istft_agrees_with_torch_istft_through_the_real_path():
    n_fft, hop = 1024, 256
    generator = torch.Generator().manual_seed(1)
    real = torch.randn(1, n_fft // 2 + 1, 12, generator=generator)
    imag = torch.randn(1, n_fft // 2 + 1, 12, generator=generator)
    imag[:, 0, :] = 0.0
    imag[:, -1, :] = 0.0

    module = OLAISTFT(n_fft, hop, n_fft, center=True)
    complex_result = module(torch.complex(real, imag))
    real_result = module(real, imag=imag)
    assert torch.allclose(complex_result, real_result, atol=1e-5)
    reference = torch.istft(
        torch.complex(real, imag),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=module.window,
        center=True,
        length=complex_result.shape[-1],
    )
    assert torch.allclose(complex_result, reference, atol=1e-4), (
        complex_result - reference
    ).abs().max()


def test_streaming_push_still_matches_offline_with_separate_parts():
    """The streaming path shares this code, and round 12 measured that it must stay identical."""
    from parakeet.audio.istft import StreamingOLA

    n_fft, hop = 512, 128
    generator = torch.Generator().manual_seed(2)
    real = torch.randn(1, n_fft // 2 + 1, 9, generator=generator)
    imag = torch.randn(1, n_fft // 2 + 1, 9, generator=generator)
    imag[:, 0, :] = 0.0
    imag[:, -1, :] = 0.0

    st = StreamingOLA(n_fft, hop, n_fft, center=False)
    pushed = st.push(real, imag=imag)
    tail = st.finalize()
    offline = OLAISTFT(n_fft, hop, n_fft, center=False)(torch.complex(real, imag))
    joined = torch.cat([pushed, tail], dim=-1)[..., : offline.shape[-1]]
    assert torch.allclose(joined, offline, atol=1e-4)
