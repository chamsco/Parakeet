"""Evaluation: dependency-free diagnostics + optional UTMOS/WER/SECS."""

from .metrics import (  # noqa: F401
    EvalReport,
    OptionalMetric,
    RTFResult,
    buzz,
    evaluate_pair,
    list_metric_availability,
    log_mel_l1,
    measure_rtf,
    mel_cepstral_distortion,
    segmental_snr,
    spectral_convergence,
    speaker_similarity,
    utmos,
    whisper_wer,
)
from .probes import (  # noqa: F401
    ae_reconstruction_l1,
    duration_error_frames,
    end_to_end_mel_l1,
    latent_normalizer_summary,
    teacher_signal_loss,
)

__all__ = [
    "EvalReport",
    "OptionalMetric",
    "RTFResult",
    "evaluate_pair",
    "measure_rtf",
    "list_metric_availability",
    "spectral_convergence",
    "log_mel_l1",
    "mel_cepstral_distortion",
    "segmental_snr",
    "buzz",
    "utmos",
    "whisper_wer",
    "speaker_similarity",
    "ae_reconstruction_l1",
    "teacher_signal_loss",
    "end_to_end_mel_l1",
    "duration_error_frames",
    "latent_normalizer_summary",
]
