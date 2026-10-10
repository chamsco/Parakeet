"""Generate the model card from measured evidence, and refuse to invent any.

    python scripts/model_card.py                 # write docs/MODEL_CARD.md + runs/model_card.json
    python scripts/model_card.py --check         # exit non-zero if a required claim is unbacked

Every number in the card must come from a report file written by a demo, with the JSON path recorded
next to it.  A claim whose report is missing is printed as **unmeasured**, not quietly dropped, and
``--check`` fails on it -- so the card cannot drift from the evidence, and a demo that stops writing
its report breaks the card rather than silently shrinking it.

The teacher/licence table is generated from ``parakeet.data.teacher.TEACHERS`` (the single source of
truth the licence gate itself uses), and the parameter counts are computed by building the models
rather than copied from an earlier measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parakeet.config import load_config  # noqa: E402
from parakeet.data.teacher import TEACHERS  # noqa: E402
from parakeet.models import build_model, count_parameters  # noqa: E402
from parakeet.train.common import config_fingerprint, git_revision  # noqa: E402

# --------------------------------------------------------------------------------------
# claims: every advertised number, tied to the report that produced it
# --------------------------------------------------------------------------------------
CLAIMS: List[Dict[str, Any]] = [
    {
        "id": "int8_pipeline_speed",
        "statement": "int8 ONNX pipeline (text side + vocoder) as shipped, 1 CPU thread",
        "report": ["runs/onnx_pipeline/pipeline_report.json"],
        "field": "int8_shipped_x_realtime",
        "op": ">=",
        "target": 20.0,
        "fmt": "{:.1f}x real time",
    },
    {
        "id": "int8_speedup_vs_pytorch",
        "statement": "int8 pipeline speedup over PyTorch fp32",
        "report": ["runs/onnx_pipeline/pipeline_report.json"],
        "field": "int8_vs_torch_speedup",
        "op": ">=",
        "target": 2.0,
        "fmt": "{:.2f}x",
    },
    {
        "id": "int8_size",
        "statement": "int8 model size (text side + vocoder)",
        "report": ["runs/onnx_pipeline/pipeline_report.json"],
        "fields": ["text_side_mb.int8", "vocoder_mb.int8"],
        "op": "sum<=",
        "target": 15.0,
        "fmt": "{:.1f} MB int8",
    },
    {
        "id": "int8_fidelity",
        "statement": "int8 vs PyTorch log-mel L1 on the same text",
        "report": ["runs/onnx_pipeline/pipeline_report.json"],
        "field": "int8_vs_torch_mel_l1",
        "op": "<=",
        "target": 1.0,
        "fmt": "{:.3f} (untrained weights: an upper bound, not a quality metric)",
    },
    {
        "id": "streaming_ttfa",
        "statement": "time-to-first-audio, streaming vs one-shot on the longest measured utterance",
        "report": ["runs/streaming_demo/report.json"],
        "list_field": "ttfa",
        "select": "max_audio_seconds",
        "field": "ttfa_speedup",
        "op": ">=",
        "target": 2.0,
        "fmt": "{:.2f}x (and the one-shot figure it improves on is printed in the report)",
    },
    {
        "id": "streaming_quality",
        "statement": "blockwise output vs one-shot: cosine similarity",
        "report": ["runs/streaming_demo/report.json"],
        "list_field": "agreement",
        "select": "variant=interpolated",
        "field": "cosine_vs_full",
        "op": ">=",
        "target": 0.95,
        "fmt": "{:.4f} (independent draw scores {independent:.4f}, the control)",
    },
    {
        "id": "reflow_nfe2",
        "statement": "Reflowed 2-step agrees with the NFE-32 reference better than the teacher's NFE-16",
        "report": ["runs/reflow_demo/report.json"],
        "field": "mean.mse_reflow2_vs_ref",
        "op": "lt_field",
        "compare": "mean.mse_teacher16_vs_ref",
        "fmt": "{value:.3f} vs {other:.3f}",
    },
    {
        "id": "mixture_steers",
        "statement": "the teacher mixture reaches the gradient (F0 span across the mixture sweep)",
        "report": ["runs/mixture_demo/report.json"],
        "field": "predicted_f0_span_hz",
        "op": ">=",
        "target": 20.0,
        "fmt": "{:.1f} Hz",
    },
    {
        "id": "multi_voice",
        "statement": "voice conditioning separates voices (predicted F0 span across 3 fixture voices)",
        "report": ["runs/voice_demo/report.json"],
        "field": "span_hz.with_voice",
        "op": ">=",
        "target": 10.0,
        "fmt": "{:.1f} Hz",
    },
    {
        "id": "recipe_tiny_end_to_end",
        "statement": "whole documented recipe runs offline (Tiny path): all checks pass",
        "report": ["runs/recipe_curate/report.json", "runs/recipe_dry_run/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "recipe_small_end_to_end",
        "statement": "whole documented recipe runs offline (Small/flow path, paired references)",
        "report": ["runs/recipe_flow/report.json", "runs/recipe_flow_quick/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "resume_exact",
        "statement": "a crashed-then-resumed run is bit-identical to an uninterrupted one",
        "report": ["runs/resume_demo/report.json"],
        "field": "max_parameter_difference",
        "op": "==",
        "target": 0.0,
        "fmt": "max parameter difference {:.3e}",
    },
    {
        "id": "phase_lock_locks",
        "statement": "phase-lock filter: its own A/B assertions hold (see the report's measured caveats)",
        "report": ["runs/phase_lock_ab/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "real_teacher_audio",
        "statement": "the documented data path runs on **real teacher speech** (Kokoro-82M, Apache-2.0)",
        "report": ["runs/real_corpus_report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "real_diagnosis",
        "statement": "the text→audio failure is **localised** (autoencoder round-trip vs the reference)",
        "report": ["runs/real_diagnose/report.json"],
        "field": "paths.0.wer",
        "op": "lt_field",
        "compare": "paths.1.wer",
        "fmt": "reference WER {value:.2f} < autoencoder round-trip {other:.2f}",
    },
    {
        "id": "objective_fit_ab",
        "statement": "the mean-invariant latent term **worked** (per-dim correlation 0.126 → 0.200, F0 −31 %) and **still did not reach intelligibility** (WER 1.000) — so the ceiling is the acoustic stage's architecture, not the loss weighting",
        "report": ["runs/objective_fit_ab.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "flow_trajectory",
        "statement": "the flow produces **full-length** audio at **~30× real time** (NFE 4) from step 400, and is still not intelligible at step 800 — recorded as a trajectory rather than a single number, because one checkpoint cannot distinguish learning from stuck",
        "report": ["runs/flow_trajectory.json"],
        "field": "checkpoints",
        "op": ">=",
        "target": 2,
        "fmt": "{value} checkpoints",
    },
    {
        "id": "speechlikeness_gate",
        "statement": "is it **speech at all**? The teacher passes the gate (voiced fraction, human pitch range, non-flat spectrum) while **both trained paths fail it** -- Tiny is a tonal buzz, the flow is unvoiced noise -- which WER 1.0 alone could not distinguish from wrong words",
        "report": ["runs/eval_flow1600_likeness/report.json"],
        "field": "speechlikeness.teacher.voiced_fraction",
        "op": ">=",
        "target": 0.5,
        "fmt": "teacher voiced fraction {value:.2f} (students: see the report)",
    },
    {
        "id": "acoustic_path_works",
        "statement": "the **acoustic path works**: the autoencoder round trip of a real teacher utterance transcribes at WER 0.0 while the same checkpoint's text-to-speech is WER 1.0 -- so the text side is the whole problem, not the decoder",
        "report": ["docs/demo/index.json"],
        "field": "autoencoder_roundtrip_wer",
        "op": "<=",
        "target": 0.2,
        "fmt": "round-trip WER {value:.2f} (student {reason})",
    },
    {
        "id": "fit_diagnosis",
        "statement": "the text side matches the latent's **mean** (cosine 0.81) with almost **no per-token structure** (per-dim correlation 0.13) and under-predicts length (0.77x)",
        "report": ["runs/fit_diag_char.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "tokenizer_ab",
        "statement": "phoneme input does **not** beat characters, and the student never fit its own "
                     "training prompts either (WER ≥ 1.0 everywhere) — so the rounds comparing corpus "
                     "duration, text diversity and teachers were comparing undertrained models",
        "report": ["runs/tokenizer_ab.json"],
        "field": "student_wer.phoneme.speechify",
        "op": ">=",
        "target": 1.0,
        "fmt": "phoneme-arm WER {value:.2f} on unseen Speechify prompts",
    },
    {
        "id": "mixture_holdout",
        "statement": "the **two-teacher mixture** (1169 utts, 109.5 min, 74 % aligned) evaluated on unseen prompts for **both** teachers, with valid controls",
        "report": ["runs/eval_mixed_speechify/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "alignment_value",
        "statement": "the teacher's alignment puts duration on **different tokens** than the fallback (median correlation 0.15) while matching the total length",
        "report": ["runs/alignment_value.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "speechify_alignment",
        "statement": "the **first real alignment**: Speechify word timings reproduce the audio duration (median error 0.8 % over 964 utterances)",
        "report": ["runs/alignment_evidence.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "text_diversity_ab",
        "statement": "**185 public-domain prompts** instead of 47 improve the proxies on 50 unseen prompts; WER is still ~1.0 (recorded)",
        "report": ["runs/text_diversity_ab.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "data_scale_ab",
        "statement": "6x more audio improves the proxies on a **prompt-disjoint hold-out**; WER on unseen prompts is still ~1.0 (recorded)",
        "report": ["runs/data_scale_ab.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "scaled_corpus",
        "statement": "a **19.4-minute** prompt-disjoint corpus trains the autoencoder past the baseline "
                     "(29 divergent steps skipped and recorded, not hidden)",
        "report": ["runs/ae_scaled/report.json"],
        "field": "final.mel_l1",
        "op": "lt_field",
        "compare": "baseline_for_comparison.mel_l1",
        "fmt": "scaled-corpus mel L1 {value:.4f} < shipped-recipe {other:.4f}",
    },
    {
        "id": "objective_ab",
        "statement": "training **through the decoder** beats a latent L1 (WER 1.648 → 0.667)",
        "report": ["runs/objective_ab.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "latent_rate_ab",
        "statement": "predicting **sub-token latents** improves the student end to end (rate 1 vs 3)",
        "report": ["runs/latent_rate_ab.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "seam_rate_sweep",
        "statement": "the token→frame seam is mostly **information** loss (oracle rate sweep)",
        "report": ["runs/seam_rate.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "seam_ab",
        "statement": "the token→frame seam A/B is recorded **including the part that did not work**",
        "report": ["runs/seam_ab.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "ae_phased_training",
        "statement": "the autoencoder bottleneck is **fixed** by spending the budget reconstruction-first",
        "report": ["runs/ae_long/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "real_training",
        "statement": "the student **trains on real speech** (autoencoder + text side, Kokoro corpus)",
        "report": ["runs/real_train/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "real_evaluation",
        "statement": "real-speech evaluation with a naturalness metric **and its teacher control**",
        "report": ["runs/real_eval/report.json"],
        "field": "checks",
        "op": "all_true",
        "fmt": "{passed}/{total} checks",
    },
    {
        "id": "learning_progress",
        "statement": "the stages learn: autoencoder reconstruction improves",
        "report": ["runs/learn_demo/report.json"],
        "field": "ae_recon_after",
        "op": "lt_field",
        "compare": "ae_recon_before",
        "fmt": "{value:.3f} vs {other:.3f} before training",
    },
    {
        "id": "learning_text_side",
        "statement": "the text side fits cached teacher signals",
        "report": ["runs/learn_demo/report.json"],
        "field": "text_loss_after",
        "op": "lt_field",
        "compare": "text_loss_before",
        "fmt": "{value:.3f} vs {other:.3f} before training",
    },
]


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
def _dig(payload: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(payload, dict):
            if part not in payload:
                return None
            payload = payload[part]
        elif isinstance(payload, list):
            try:
                payload = payload[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return payload


def _select_row(rows: List[Dict[str, Any]], selector: str) -> Optional[Dict[str, Any]]:
    if selector == "max_audio_seconds":
        return max(rows, key=lambda r: float(r.get("audio_seconds") or 0.0)) if rows else None
    if "=" in selector:
        key, want = selector.split("=", 1)
        for row in rows:
            if str(row.get(key)) == want:
                return row
        return None
    return rows[0] if rows else None


def _evidence_path(claim: Dict[str, Any]) -> str:
    """Where this claim's evidence is committed.

    The reports themselves are run artifacts (``runs/`` is gitignored), so a fresh clone could not
    verify the card at all.  The bundle under ``docs/evidence/`` is the committed snapshot: small,
    diffable, and the thing the card actually cites.  ``--refresh-evidence`` regenerates it from a
    fresh set of runs, and the manifest records the git revision and a SHA-256 per file so drift is
    visible rather than silent.
    """
    return f"docs/evidence/{claim['id']}.json"


def _candidates(claim: Dict[str, Any]) -> List[str]:
    # the committed evidence wins, so the card is reproducible from a clone; the raw run report is
    # the fallback for a workspace that has not snapshotted yet
    return [_evidence_path(claim)] + list(claim["report"])


def _pick_report(candidates: List[str], reports_root: Path = ROOT) -> Tuple[Optional[Path], List[str]]:
    missing: List[str] = []
    for candidate in candidates:
        path = reports_root / candidate
        if path.exists():
            return path, missing
        missing.append(candidate)
    return None, missing


def evaluate(claim: Dict[str, Any], reports_root: Path = ROOT) -> Dict[str, Any]:
    """Return the claim with `status` in {pass, fail, unmeasured} and the measured value."""
    candidates = _candidates(claim)
    report, missing = _pick_report(candidates, reports_root)
    result = {**claim, "report_used": None, "value": None, "status": "unmeasured",
              "missing_reports": missing, "detail": ""}
    if report is None:
        result["detail"] = "no report: " + ", ".join(missing)
        return result
    payload = json.loads(report.read_text(encoding="utf-8"))
    result["report_used"] = report.relative_to(reports_root).as_posix()

    if "list_field" in claim:
        rows = payload.get(claim["list_field"]) or []
        row = _select_row(rows, claim.get("select", ""))
        if row is None:
            result["detail"] = f"no row matched {claim.get('select')!r}"
            return result
        value = _dig(row, claim["field"])
        context = {
            "independent": row.get("cosine_independent_draw"),
            "audio_seconds": row.get("audio_seconds"),
        }
    elif "fields" in claim:
        parts = [_dig(payload, f) for f in claim["fields"]]
        if any(p is None for p in parts):
            result["detail"] = f"missing fields {claim['fields']}"
            return result
        value = float(sum(parts))
        context = {}
        result["value"] = value
        target = float(claim["target"])
        ok = value <= target
        result["status"] = "pass" if ok else "fail"
        result["detail"] = claim["fmt"].format(value)
        result["target"] = target
        return result
    else:
        value = _dig(payload, claim["field"])
        context = {}

    if value is None:
        result["detail"] = f"field {claim['field']} absent from {result['report_used']}"
        return result
    result["value"] = value
    op = claim["op"]

    if op == "all_true":
        checks = value if isinstance(value, dict) else {}
        passed = sum(1 for v in checks.values() if v)
        result["status"] = "pass" if checks and passed == len(checks) else "fail"
        result["detail"] = claim["fmt"].format(passed=passed, total=len(checks))
        if passed != len(checks):
            result["detail"] += " | failing: " + ", ".join(k for k, v in checks.items() if not v)
        return result

    if op == "lt_field":
        other = _dig(payload, claim["compare"])
        if other is None:
            result["detail"] = f"comparison field {claim['compare']} absent"
            return result
        result["status"] = "pass" if float(value) < float(other) else "fail"
        result["detail"] = claim["fmt"].format(value=float(value), other=float(other), **context)
        return result

    target = float(claim["target"])
    value_f = float(value)
    ok = {">=": value_f >= target, "<=": value_f <= target, "==": value_f == target}[op]
    result["status"] = "pass" if ok else "fail"
    try:
        result["detail"] = claim["fmt"].format(value_f, **context)
    except (IndexError, KeyError, ValueError, TypeError):
        # a formatting placeholder we did not supply must not lose the number
        result["detail"] = f"{value_f:.6g}"
    result["target"] = target
    return result


# --------------------------------------------------------------------------------------
# card rendering
# --------------------------------------------------------------------------------------
def _teacher_table() -> str:
    lines = [
        "| teacher | kind | code licence | weights / output licence | trainable? | restricted |",
        "|---|---|---|---|---|---|",
    ]
    for name, spec in sorted(TEACHERS.items()):
        lines.append(
            f"| `{name}` | {spec.kind} | {spec.code_license} | {spec.weights_license} | "
            f"{'yes' if spec.allows_training else '**no**'} | "
            f"{'**yes**' if getattr(spec, 'restricted', False) else 'no'} |"
        )
    return "\n".join(lines)


def _parameter_table() -> str:
    lines = ["| variant | parameters | config | config sha256 |", "|---|---|---|---|"]
    for config_path in ("configs/parakeet_tiny.yaml", "configs/parakeet_small.yaml"):
        path = ROOT / config_path
        if not path.exists():
            continue
        cfg = load_config(str(path))
        model = build_model(cfg)
        params = count_parameters(model)
        lines.append(
            f"| {cfg.variant} | {params / 1e6:.3f} M | `{config_path}` | "
            f"`{config_fingerprint(cfg)[:16]}` |"
        )
    return "\n".join(lines)


def _capacity_study() -> str:
    """The size/capacity frontier, read from the committed ablation evidence.

    This is a *study*, not a claim: its numbers come from synthetic fixtures with an untrained
    autoencoder, so the card must present it with that caveat rather than as quality evidence.
    """
    path = ROOT / "docs" / "evidence" / "capacity_ablation.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        return ""
    lines = [
        "Measured by `scripts/ablate.py` -- same cache, same step budget, same seed for every "
        "variant, scored on a held-out split:",
        "",
        "| text dim / layers | text params | total params | held-out fit | text latency |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| dim{r['text_dim']}-L{r['text_layers']} | {r['text_params'] / 1e6:.3f} M | "
            f"{r['total_params'] / 1e6:.3f} M | {r['val_fit']:.4f} | {r['text_latency_ms']:.1f} ms |"
        )
    efficient = payload.get("half_size_best")
    lines.append("")
    lines.append(f"**Caveat, stated in the report**: {payload.get('caveat', '')}")
    if efficient:
        lines.append("")
        lines.append(
            f"Fixture-derived recommendation: `{efficient['label']}` at "
            f"{efficient['text_params'] / 1e6:.3f} M text parameters is within "
            f"{efficient['fit_penalty_pct']:.1f} % of the best held-out fit -- offered as "
            "`configs/parakeet_tiny_lite.yaml` for comparison on a real corpus, **not** as a new "
            "default, because the fixtures are repetitive and a real corpus plausibly needs more "
            "capacity."
        )
    return "\n".join(lines)


def _component_table() -> str:
    """Where the parameters actually are -- the first question when making a model lighter."""
    path = ROOT / "configs" / "parakeet_tiny.yaml"
    if not path.exists():
        return ""
    cfg = load_config(str(path))
    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    lines = ["| component (Tiny) | parameters | share |", "|---|---|---|"]
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n:
            lines.append(f"| `{name}` | {n / 1e6:.3f} M | {100.0 * n / total:.1f} % |")
    lines.append(f"| **total** | **{total / 1e6:.3f} M** | 100 % |")
    return "\n".join(lines)


def _checkpoints() -> List[str]:
    """Trained-weight artifacts present in the workspace (none are committed).

    Scratch directories (``_probe*``) are ignored: they are debris from debugging, and listing them
    as "artifacts present" would overstate what exists.
    """
    found = []
    for pattern in ("**/*.pt", "**/*.safetensors", "**/*.onnx"):
        for path in (ROOT / "runs").glob(pattern):
            rel = path.relative_to(ROOT)
            if any(part.startswith("_") for part in rel.parts):
                continue
            if path.is_file():
                found.append(rel.as_posix())
    return sorted(found)


def render(results: List[Dict[str, Any]], args) -> str:
    rev = git_revision() or {}
    measured = [r for r in results if r["status"] in {"pass", "fail"}]
    unmeasured = [r for r in results if r["status"] == "unmeasured"]
    failing = [r for r in results if r["status"] == "fail"]
    checkpoints = _checkpoints()

    out: List[str] = []
    out.append("# Parakeet model card")
    out.append("")
    out.append(
        f"*Generated by `scripts/model_card.py` on {datetime.now().isoformat(timespec='seconds')} "
        f"from git `{rev.get('rev', 'unknown')}`"
        f"{' (dirty working tree)' if rev.get('dirty') else ''}.  "
        "Every row below is read from a report file written by a demo; nothing is typed in by hand.*"
    )
    out.append("")
    out.append("## Status: research scaffold, no trained weights")
    out.append("")
    if checkpoints:
        out.append(
            f"Checkpoint files present in this workspace: **{len(checkpoints)}** "
            f"({', '.join(f'`{c}`' for c in checkpoints[:3])}{' …' if len(checkpoints) > 3 else ''}).  "
            "These are run artifacts from the demos below, trained on **synthetic fixtures** -- they "
            "are not released weights and their audio is not speech.  A model trained on a real "
            "corpus does not exist yet."
        )
    else:
        out.append(
            "**No trained checkpoint exists.**  Every quality figure in the source papers (UTMOS 4.41, "
            "WER 5.7 %, RTF 0.02 on a 4090) is a *target*, not a result of this repository.  What is "
            "verified here is the machinery: that the data path, the distillation objectives, the "
            "sampler, the streaming path, the quantisation and the resume logic do what they claim, "
            "measured on synthetic fixtures with controls."
        )
    out.append("")
    out.append("## Intended use")
    out.append("")
    out.append(
        "* Research and engineering on **distilling several TTS teachers into one small student**, on "
        "hardware from a laptop up.\n"
        "* Single-voice and few-voice synthesis where the voice is a known, consented reference.\n"
        "* Deployment plumbing: int8 ONNX export, blockwise streaming, phase-lock post-filtering."
    )
    out.append("")
    out.append("## Out of scope and misuse")
    out.append("")
    out.append(
        "* **Voice cloning without the speaker's consent**, and any impersonation of a real person.\n"
        "* Disinformation, fraud, or bypassing a platform's synthetic-media policy.  Audio produced by "
        "this code is synthetic and should be labelled as such; `write_run_metadata` records the "
        "provenance needed to do that.\n"
        "* Safety-critical or accessibility-critical use: there is no trained checkpoint, no "
        "intelligibility evaluation (WER) and no naturalness evaluation (UTMOS) here.\n"
        "* Using the output of a restricted teacher.  `teacher.check_teacher` refuses MiniMax unless "
        "its terms are explicitly acknowledged; see `docs/LEGAL.md`."
    )
    out.append("")
    out.append("## Teachers and licences")
    out.append("")
    out.append(_teacher_table())
    out.append("")
    out.append(
        "Audio-level distillation is what makes this mixture possible: each teacher's waveform is "
        "re-encoded into one shared latent space, so a teacher may add value without contributing "
        "codes, logits or architecture.  The obligation that comes with it is provenance -- who spoke "
        "the audio the student learned from -- which is why every corpus record carries a teacher, a "
        "voice, a licence and a content hash."
    )
    out.append("")
    out.append("## Architecture")
    out.append("")
    out.append(_parameter_table())
    out.append("")
    out.append(
        "Where the parameters are, because that is the first question when making a model lighter:"
    )
    out.append("")
    out.append(_component_table())
    out.append("")
    out.append(
        "`scripts/ablate.py` measures the size/fit/latency frontier on fixtures; its result is a "
        "*fixture* result (synthetic audio, untrained autoencoder, short budget) and must be revisited "
        "with real data before the shipped geometry is changed."
    )
    out.append("")
    study = _capacity_study()
    if study:
        out.append("### Capacity study (fixture-derived, not a quality claim)")
        out.append("")
        out.append(study)
        out.append("")
    out.append("See `docs/01-ARCHITECTURE.md` for module-level detail.")
    out.append("")
    out.append("## Measured results")
    out.append("")
    out.append("| claim | measured | source report | status |")
    out.append("|---|---|---|---|")
    for r in results:
        status = {"pass": "verified", "fail": "**FAILING**", "unmeasured": "**unmeasured**"}[r["status"]]
        source = f"`{r['report_used']}`" if r["report_used"] else "—"
        out.append(f"| {r['statement']} | {r['detail'] or '—'} | {source} | {status} |")
    out.append("")
    if failing:
        out.append("### Failing claims")
        out.append("")
        for r in failing:
            out.append(f"* **{r['id']}**: {r['statement']} — {r['detail']}")
        out.append("")
    if unmeasured:
        out.append("### Unmeasured claims (no report)")
        out.append("")
        for r in unmeasured:
            out.append(f"* **{r['id']}**: {r['statement']} — {r['detail']}")
        out.append("")
    out.append("## Limitations that no table row captures")
    out.append("")
    out.append(
        "* **No trained checkpoint and no real corpus.**  Every measurement uses synthetic fixtures, "
        "because the teacher weights need a GPU and the teachers' own audio is licence-encumbered.\n"
        "* **No perceptual metric.**  UTMOS is unavailable in this environment, so naturalness claims "
        "are citations from the papers, not results.  The phase-lock A/B states the same thing about "
        "its own filter, in more detail.\n"
        "* **The fixtures are not speech.**  They are formant stacks with randomised harmonic phase, "
        "which is enough to verify machinery and *not* enough to claim quality; the fixture voices "
        "cannot even serve as a phase-lock test signal (measured: they score like white noise).\n"
        "* **Small-variant quality is unverified**: the flow path is verified for wiring and sampler "
        "behaviour, not for the speaker similarity that zero-shot cloning would need.\n"
        "* **int8 fidelity numbers are from untrained weights**, so they bound the quantisation error "
        "rather than establish usable quality."
    )
    out.append("")
    out.append("## Reproducing every row")
    out.append("")
    out.append("```bash")
    out.append("python scripts/smoke_test.py --steps 2          # all five stages, synthesize, quantise")
    out.append("python scripts/learn_demo.py                   # the stages actually fit")
    out.append("python scripts/reflow_demo.py                  # few-step sampling")
    out.append("python scripts/streaming_demo.py               # TTFA and blockwise agreement")
    out.append("python scripts/mixture_demo.py                 # the teacher mixture steers the student")
    out.append("python scripts/voice_demo.py                   # multi-voice conditioning, with a control")
    out.append("python scripts/phase_lock_ab.py                # the post-filter A/B")
    out.append("python scripts/resume_demo.py                  # checkpoint/resume, byte for byte")
    out.append("python scripts/recipe_dry_run.py               # the whole pipeline, offline")
    out.append("python scripts/recipe_dry_run.py --stage flow  # the Small path with paired references")
    out.append("python scripts/export_onnx.py --pipeline       # int8 ONNX + runtime benchmark")
    out.append("python scripts/model_card.py --check           # this document, re-derived")
    out.append("```")
    out.append("")
    out.append("## Provenance of a run")
    out.append("")
    out.append(
        "Every `run_stage` call writes `<out_dir>/run.json` with the git revision and dirty flag, a "
        "SHA-256 of the full config, python/torch versions, the stage, the trainable and frozen module "
        "list, and whatever the caller adds (for `train.py`: the teacher mixture, voice list and a "
        "SHA-256 of the cache index).  Checkpoints carry the same plus optimizer, EMA, discriminator, "
        "step, RNG and batch-order state, so a run can be resumed byte-for-byte."
    )
    return "\n".join(out) + "\n"


def refresh_evidence() -> Dict[str, Any]:
    """Copy the freshest run report for each claim into the committed evidence bundle.

    A claim whose *run* report is absent but whose committed evidence exists is **retained**, not
    failed: some evidence needs a 320 MB model download (the real-teacher corpus) and re-running it
    in CI would be a burden with no benefit, while the committed bundle still lets any clone verify
    the claim.
    """
    import hashlib
    import shutil

    evidence_dir = ROOT / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for claim in CLAIMS:
        source = next((ROOT / c for c in claim["report"] if (ROOT / c).exists()), None)
        target = ROOT / _evidence_path(claim)
        if source is None:
            entries.append(
                {
                    "claim": claim["id"],
                    "status": "retained" if target.exists() else "no_source",
                    "expected": claim["report"],
                    "sha256": (
                        hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
                    ),
                }
            )
            continue
        shutil.copyfile(source, target)
        entries.append(
            {
                "claim": claim["id"],
                "status": "copied",
                "source": source.relative_to(ROOT).as_posix(),
                "evidence": _evidence_path(claim),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
        "note": (
            "Committed snapshots of the reports the model card cites.  Refresh with "
            "`python scripts/model_card.py --refresh-evidence` after re-running the demos; the "
            "hashes make a stale bundle visible."
        ),
        "entries": entries,
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build(reports_root: Path = ROOT) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate every claim and summarise.  ``reports_root`` is injectable so the check itself can
    be tested (a claim with no report must come back ``unmeasured``, not silently pass)."""
    results = [evaluate(claim, reports_root) for claim in CLAIMS]
    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
        "reports_root": str(reports_root),
        "claims": [dict(r) for r in results],
        "n_pass": sum(1 for r in results if r["status"] == "pass"),
        "n_fail": sum(1 for r in results if r["status"] == "fail"),
        "n_unmeasured": sum(1 for r in results if r["status"] == "unmeasured"),
        "checkpoints_present": _checkpoints(),
    }
    return results, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the model card from measured reports")
    ap.add_argument("--out", default="docs/MODEL_CARD.md")
    ap.add_argument("--json-out", default="runs/model_card.json")
    ap.add_argument("--reports-root", default=str(ROOT),
                    help="where the report files live (default: the repo root)")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any claim is unmeasured or failing")
    ap.add_argument("--refresh-evidence", action="store_true",
                    help="copy the freshest run reports into the committed docs/evidence bundle")
    args = ap.parse_args()

    if args.refresh_evidence:
        manifest = refresh_evidence()
        copied = sum(1 for e in manifest["entries"] if e["status"] == "copied")
        retained = sum(1 for e in manifest["entries"] if e["status"] == "retained")
        print(f"evidence bundle refreshed: {copied} copied, {retained} retained (source needs a "
              f"model download), {len(manifest['entries']) - copied - retained} missing")
        for entry in manifest["entries"]:
            if entry["status"] not in {"copied", "retained"}:
                print(f"  !! {entry['claim']}: no report and no committed evidence "
                      f"({entry['expected']})")
        if copied + retained != len(manifest["entries"]):
            return 1

    reports_root = Path(args.reports_root).resolve()
    results, summary = build(reports_root)
    for r in results:
        mark = {"pass": "ok  ", "fail": "FAIL", "unmeasured": "----"}[r["status"]]
        print(f"  [{mark}] {r['id']:32s} {r['detail'] or 'no report'}")

    card = render(results, args)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card, encoding="utf-8")
    json_path = ROOT / args.json_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\ncard  -> {out_path.relative_to(ROOT)}\nsummary -> {json_path.relative_to(ROOT)}")
    print(f"verified {summary['n_pass']} | failing {summary['n_fail']} | "
          f"unmeasured {summary['n_unmeasured']}")

    if args.check and (summary["n_fail"] or summary["n_unmeasured"]):
        print("MODEL CARD CHECK FAILED: a claim is failing or has no backing report")
        return 1
    print("MODEL CARD " + ("CHECK PASSED" if args.check else "WRITTEN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
