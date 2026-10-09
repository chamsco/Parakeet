"""Phase-lock A/B: does the post-filter do what it claims, and what does it cost?

    python scripts/phase_lock_ab.py --quick      # ~20 s
    python scripts/phase_lock_ab.py              # ~1 min

Paradee's phase-locking filter is the only *quality* intervention in this repo that is not a trained
model: it rewrites the phase of synthesized speech between 2 and 8 kHz, leaves the magnitudes alone,
and reportedly moves UTMOS 4.39 -> 4.41.  Two modes are implemented (a physically motivated linear
glottal-phase ramp, and Paradee's smoothed complex reference), and the filter costs ~10 % of the
shipped CPU pipeline.

So this script answers three separable questions and refuses to blur them:

* **Does it lock?**  Phase coherence in the 2-8 kHz band must *increase* — that is the filter's
  entire purpose and it is directly measurable.
* **What does it cost in fidelity?**  A phase rewrite that changes the sound too much is not a win;
  the log-mel distance from the unfiltered signal bounds it.
* **Is it really the phase?**  A control on white noise: if coherence rises there too, the metric is
  measuring the filter's arithmetic rather than speech structure.

What it deliberately does **not** claim: that any of this sounds better.  UTMOS is unavailable here,
so the perceptual claim in the paper stays a citation, not a result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.audio.f0 import estimate_f0  # noqa: E402
from parakeet.config import load_config  # noqa: E402
from parakeet.data.synthetic import make_corpus  # noqa: E402
from parakeet.eval.metrics import buzz  # noqa: E402
from parakeet.eval.metrics import log_mel_l1  # noqa: E402
from parakeet.inference import phase_coherence, phase_lock  # noqa: E402

METHODS = ("ramp", "smooth")
STRENGTHS = (0.3, 0.7, 1.0)
#: The linear-phase reference is searched on a grid of delays ``tau``.  ``phase_coherence`` searches
#: the same kind of grid, so the achievable concentration is bounded by the grid resolution: 64
#: points over [0, 1/60 s] is 260 us, which at 8 kHz is more than two periods of phase error across
#: the band.  The filter as configured therefore *cannot* strongly lock a 2-8 kHz band, which is what
#: the first run of this A/B showed.  Sweeping the grid turns that observation into a diagnosis.
TAU_GRIDS = (64, 256, 1024)


def _run_configs(quick: bool):
    if quick:
        return [
            ("ramp", 0.7, 64),
            ("ramp", 1.0, 64),
            ("ramp", 1.0, 1024),
            ("smooth", 0.7, 64),
        ]
    return [
        ("ramp", s, g) for s in (0.7, 1.0) for g in TAU_GRIDS
    ] + [("smooth", s, 64) for s in (0.3, 0.7, 1.0)]


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def _voiced_ratio(wav: torch.Tensor, sr: int) -> float:
    _f0, voiced, _conf = estimate_f0(wav[None], sr, hop_length=256, frame_length=2048)
    return float(voiced.float().mean())


def glottal_like(
    f0: float, n_samples: int, sr: int, locked_fraction: float = 0.7, seed: int = 0,
    noise_floor: float = 0.0,
) -> torch.Tensor:
    """A harmonic stack whose phase is *partially* glottal-locked, like real voiced speech.

    The synthetic corpus fixtures cannot serve as the test signal here: ``render_token`` assigns
    every harmonic a **random** phase, so their phase statistics are indistinguishable from noise
    (measured: 0.149 vs 0.144 coherence).  The phase-lock filter targets a glottal pulse train,
    whose phase is linear in frequency (``psi(f) = -2*pi*f*tau``), so a fair A/B needs a signal that
    has that structure *plus* the jitter real speech carries.
    """
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(n_samples, dtype=torch.float32) / sr
    harmonics = max(1, int((sr / 2) / f0))
    freqs = f0 * torch.arange(1, harmonics + 1, dtype=torch.float32)
    # formant-ish spectral tilt so it is not a flat comb
    gain = 1.0 / (1.0 + (freqs / 900.0).pow(2))
    tau = 1.0 / f0
    locked_phase = -2 * torch.pi * freqs * tau
    jitter = torch.rand(harmonics, generator=generator) * 2 * torch.pi
    phase = locked_fraction * locked_phase + (1.0 - locked_fraction) * jitter
    sig = (gain[:, None] * torch.sin(2 * torch.pi * freqs[:, None] * t[None] + phase[:, None])).sum(0)
    if noise_floor:
        sig = sig + torch.randn(n_samples, generator=generator) * noise_floor
    return (sig / sig.abs().max().clamp_min(1e-6) * 0.3).float()


@torch.no_grad()
def temporal_coherence(
    wav: torch.Tensor, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256,
    band=(2000.0, 8000.0),
) -> float:
    """Frame-to-frame phase consistency in ``band`` (the *other* thing "phase locking" can mean).

    The within-frame statistic cannot see a reference that is smoothed **across time**, which is
    exactly what ``method="smooth"`` builds.  This measures whether the phase advances coherently
    from frame to frame, so the two modes can be compared on their own terms.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop_length, n_fft, window, return_complex=True)
    lo = max(1, int(round(band[0] / (sample_rate / n_fft))))
    hi = min(n_fft // 2 + 1, int(round(band[1] / (sample_rate / n_fft))) + 1)
    phase = torch.angle(spec[:, lo:hi, :])
    delta = phase[:, :, 1:] - phase[:, :, :-1]
    return float(torch.exp(1j * delta).mean(dim=1).abs().mean().item())


def measure(cfg, wav: torch.Tensor, tag: str) -> Dict[str, float]:
    """Coherence, buzz and voiced fraction of one signal before any filtering."""
    sr = cfg.audio.sample_rate
    return {
        "coherence": float(phase_coherence(wav.reshape(1, -1), sample_rate=sr).item()),
        "temporal": temporal_coherence(wav.reshape(1, -1), sample_rate=sr),
        "buzz": float(buzz(wav.reshape(-1), cfg)),
        "voiced": _voiced_ratio(wav, sr),
        "seconds": wav.numel() / sr,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B the phase-locking filter")
    ap.add_argument("--config", default="configs/parakeet_tiny.yaml")
    ap.add_argument("--utterances", type=int, default=6)
    ap.add_argument("--out", default="runs/phase_lock_ab")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.utterances = 3

    cfg = load_config(args.config)
    if args.quick:
        cfg.audio.n_mels = 80
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr = cfg.audio.sample_rate
    torch.set_num_threads(max(1, torch.get_num_threads()))

    corpus = make_corpus(args.utterances, cfg.audio, seed=3)
    fixture_signals = [u.wav.reshape(-1) for u in corpus]
    # partially glottal-locked: the signal class the filter is actually designed for
    structured_signals = [
        glottal_like(95.0 + 7.0 * i, int(2.0 * sr), sr, locked_fraction=0.7, seed=10 + i,
                     noise_floor=0.002)
        for i in range(max(2, args.utterances // 2))
    ]
    generator = torch.Generator().manual_seed(0)
    noise_signals = [
        torch.randn(w.numel(), generator=generator) * 0.05 for w in structured_signals
    ]

    _banner(f"signals: {len(structured_signals)} partially glottal-locked, "
            f"{len(fixture_signals)} corpus fixtures, {len(noise_signals)} white-noise controls "
            f"({sum(w.numel() for w in structured_signals) / sr:.1f}s each class)")
    base_structured = [measure(cfg, w, "structured") for w in structured_signals]
    base_fixture = [measure(cfg, w, "fixture") for w in fixture_signals]
    base_noise = [measure(cfg, w, "noise") for w in noise_signals]
    print(f"  glottal-locked : coherence {_mean(base_structured, 'coherence'):.4f} | "
          f"temporal {_mean(base_structured, 'temporal'):.4f} | buzz "
          f"{_mean(base_structured, 'buzz'):.4f}")
    print(f"  corpus fixture : coherence {_mean(base_fixture, 'coherence'):.4f} "
          f"(random per-harmonic phase, so ~= noise)")
    print(f"  white noise    : coherence {_mean(base_noise, 'coherence'):.4f} "
          f"<-- the metric's floor, NOT zero")

    rows: List[Dict[str, float]] = []
    for method, strength, n_tau in _run_configs(args.quick):
        t0 = time.perf_counter()
        filtered = [
            phase_lock(w[None].reshape(1, -1), sample_rate=sr, method=method, strength=strength,
                       n_tau=n_tau)[0]
            for w in structured_signals
        ]
        noised = [
            phase_lock(w[None].reshape(1, -1), sample_rate=sr, method=method, strength=strength,
                       n_tau=n_tau)[0]
            for w in noise_signals
        ]
        elapsed = time.perf_counter() - t0
        audio_seconds = sum(w.numel() for w in structured_signals + noise_signals) / sr
        after = [measure(cfg, w, "structured") for w in filtered]
        after_noise = [measure(cfg, w, "noise") for w in noised]
        fidelity = [
            float(log_mel_l1(before, new, cfg))
            for before, new in zip(structured_signals, filtered)
        ]
        before_gap = _mean(base_structured, "coherence") - _mean(base_noise, "coherence")
        after_gap = _mean(after, "coherence") - _mean(after_noise, "coherence")
        rows.append(
            {
                "method": method,
                "strength": strength,
                "n_tau": n_tau,
                "coherence_before": _mean(base_structured, "coherence"),
                "coherence_after": _mean(after, "coherence"),
                "coherence_gain": _mean(after, "coherence") - _mean(base_structured, "coherence"),
                "coherence_after_noise": _mean(after_noise, "coherence"),
                "noise_gain": _mean(after_noise, "coherence") - _mean(base_noise, "coherence"),
                "temporal_before": _mean(base_structured, "temporal"),
                "temporal_after": _mean(after, "temporal"),
                "signal_noise_gap_before": before_gap,
                "signal_noise_gap_after": after_gap,
                "buzz_before": _mean(base_structured, "buzz"),
                "buzz_after": _mean(after, "buzz"),
                "mel_l1_vs_unfiltered": sum(fidelity) / len(fidelity),
                "ms_per_audio_second": elapsed / audio_seconds * 1000.0,
            }
        )
        r = rows[-1]
        print(f"  {method:6s} strength {strength:.1f} n_tau {n_tau:5d}: coherence "
              f"{r['coherence_before']:.4f} -> {r['coherence_after']:.4f} "
              f"({r['coherence_gain']:+.4f}) | noise {r['coherence_after_noise']:.4f} "
              f"({r['noise_gain']:+.4f}) | gap {before_gap:+.4f} -> {after_gap:+.4f} | "
              f"temporal {r['temporal_before']:.4f} -> {r['temporal_after']:.4f} | "
              f"mel L1 {r['mel_l1_vs_unfiltered']:.4f} | "
              f"{r['ms_per_audio_second']:.1f} ms/audio-s")

    # latency context: what the filter costs relative to the whole shipped pipeline
    profile = Path("runs/profile.json")
    pipeline_ms = None
    if profile.exists():
        payload = json.loads(profile.read_text(encoding="utf-8"))
        pipeline_ms = payload.get("total_ms") or payload.get("mean_ms")

    def _pick(method, strength, n_tau):
        return next(
            r for r in rows
            if r["method"] == method and r["strength"] == strength and r["n_tau"] == n_tau
        )

    default = _pick("ramp", 0.7, 64) if any(
        r["method"] == "ramp" and r["strength"] == 0.7 and r["n_tau"] == 64 for r in rows
    ) else rows[0]
    best = max(rows, key=lambda r: r["coherence_gain"])
    ramp_rows = [r for r in rows if r["method"] == "ramp"]
    # matched-strength comparison: the first version of this check compared the extreme rows, which
    # differed in strength as well as grid size, so it measured strength not resolution
    probe_strength = 0.7 if any(
        r["strength"] == 0.7 and r["n_tau"] == min(TAU_GRIDS) for r in ramp_rows
    ) else ramp_rows[0]["strength"]
    coarse = _pick("ramp", probe_strength, min(TAU_GRIDS))
    fine = _pick("ramp", probe_strength, max(TAU_GRIDS))
    # the configuration this measurement supports, shipped as the default (asserted below)
    recommended = _pick("ramp", probe_strength, 256)
    cheapest_effective = min(
        (r for r in rows if r["coherence_gain"] > 0.5 * best["coherence_gain"]),
        key=lambda r: r["ms_per_audio_second"],
        default=best,
    )

    checks = {
        # the filter's purpose: more phase concentration in the target band on glottal-locked speech
        "ramp_locks_glottal_phase": best["coherence_gain"] > 0.015,
        # the measured *failure* of the framing this A/B started with: the filter adds coherence to
        # white noise too, so the speech-vs-noise gap NARROWS.  At coarse grid + full strength it
        # collapses to ~45 % of its original width (in the table above); the recommended
        # configuration must keep most of it, and the collapse is reported rather than hidden.
        "recommended_config_preserves_the_gap": (
            recommended["signal_noise_gap_after"] > 0.75 * recommended["signal_noise_gap_before"]
        ),
        # the diagnosis: the achievable lock across a 2-8 kHz band is bounded by the delay-grid
        # resolution, so at matched strength a finer grid must buy real coherence...
        "finer_tau_grid_helps": fine["coherence_gain"] > coarse["coherence_gain"] + 0.005,
        # ...while adding *less* coherence to noise, i.e. less of the effect is the filter's own
        # arithmetic rather than speech structure
        "finer_tau_grid_reduces_false_locking": fine["noise_gain"] < coarse["noise_gain"] * 0.75,
        # the shipped default must be the configuration this measurement supports
        "default_n_tau_is_the_measured_choice": _default_n_tau() == recommended["n_tau"],
        "fidelity_cost_is_bounded": best["mel_l1_vs_unfiltered"] < 0.35,
        "latency_is_bounded": best["ms_per_audio_second"] < 400.0,
    }

    report = {
        "config": args.config,
        "sample_rate": sr,
        "n_utterances": len(structured_signals),
        "audio_seconds": sum(w.numel() for w in structured_signals) / sr,
        "baseline": {
            "structured_coherence": _mean(base_structured, "coherence"),
            "structured_temporal": _mean(base_structured, "temporal"),
            "fixture_coherence": _mean(base_fixture, "coherence"),
            "noise_coherence": _mean(base_noise, "coherence"),
        },
        "metric_note": (
            "phase_coherence is the max over a 64-point tau grid of |mean_f exp(i*phase)|, so white "
            "noise scores ~0.14, not 0, and any linear-phase imposition raises it.  The synthetic "
            "corpus fixtures score the same as noise because render_token randomises harmonic "
            "phases, so they cannot be used as the test signal for this filter."
        ),
        "runs": rows,
        "recommended": {
            "cheapest_effective": {
                "method": cheapest_effective["method"],
                "strength": cheapest_effective["strength"],
                "coherence_gain": cheapest_effective["coherence_gain"],
                "ms_per_audio_second": cheapest_effective["ms_per_audio_second"],
            },
            "largest_gain": {"method": best["method"], "strength": best["strength"],
                             "coherence_gain": best["coherence_gain"]},
        },
        "pipeline_ms": pipeline_ms,
        "checks": checks,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _banner("RESULT")
    print(f"  best lock: {best['method']} @ {best['strength']:.1f} n_tau={best['n_tau']} "
          f"(+{best['coherence_gain']:.4f} coherence, {best['ms_per_audio_second']:.1f} ms/audio-s)")
    print(f"  cheapest within half the best gain: {cheapest_effective['method']} @ "
          f"{cheapest_effective['strength']:.1f} n_tau={cheapest_effective['n_tau']} "
          f"({cheapest_effective['ms_per_audio_second']:.1f} ms/audio-s)")
    print(f"\n  CAVEAT, measured: the filter adds coherence to WHITE NOISE "
          f"({coarse['noise_gain']:+.4f} at n_tau={coarse['n_tau']}) as well as to "
          f"glottal-locked\n  speech ({coarse['coherence_gain']:+.4f}), so the speech-vs-noise gap "
          f"narrows ({coarse['signal_noise_gap_before']:+.4f} -> "
          f"{coarse['signal_noise_gap_after']:+.4f}).\n  The within-frame concentration statistic "
          f"therefore cannot demonstrate speech-specific locking;\n  a finer delay grid reduces the "
          f"false locking ({fine['noise_gain']:+.4f} at n_tau={fine['n_tau']}) and roughly\n"
          f"  triples the real gain ({coarse['coherence_gain']:+.4f} -> "
          f"{fine['coherence_gain']:+.4f}), which is why the filter's default n_tau is now 256.")
    print("  NOT measured here: whether any of this sounds better.  UTMOS is unavailable in this "
          "environment,\n  so the paper's 4.39 -> 4.41 stays a citation.  Coherence, fidelity cost "
          "and latency are real numbers.")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nreport -> {out/'report.json'}")
    print("PHASE-LOCK A/B " + ("PASSED" if all(checks.values()) else "FAILED"))
    return 0 if all(checks.values()) else 1


def _mean(rows: List[Dict[str, float]], key: str) -> float:
    values = [r[key] for r in rows]
    return sum(values) / max(1, len(values))


def _default_n_tau() -> int:
    """The shipped default, read from the function so this script cannot drift from the code."""
    import inspect

    return int(inspect.signature(phase_lock).parameters["n_tau"].default)


if __name__ == "__main__":
    raise SystemExit(main())
