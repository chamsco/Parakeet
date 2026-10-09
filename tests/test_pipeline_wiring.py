"""Pipeline wiring: the documented entry points must use the features built for them.

Rounds 6-9 wired the teacher mixture, multi-voice conditioning and cross-sample pairing into the
*library*. Round 10 found that both CLI entry points bypassed all three:

* ``scripts/train.py`` built ``LatentShardBatchSource`` without ``pair_references``, so the
  documented ``--stage flow`` command trained on each utterance's own mel -- no cross-sample
  pairing at all -- and without any reference-length cap;
* ``scripts/make_teacher_corpus.py --cache-only`` called ``build_latent_cache`` **without** a
  mixture, so every teacher weight became 1.0 and the mixture reverted to a decoration on the
  documented path;
* ``curate_manifest`` (the whole documented P1 quality pipeline) was **dead code**: no entry point
  called it.

The fix is structural: the decisions moved into ``make_batch_source`` and
``cache_teacher_corpus``, which is what these tests exercise.
"""

import copy
import json

import pytest
import torch

from parakeet.data.curate import CurateConfig, curate_manifest
from parakeet.data.dataset import LatentShardBatchSource, LatentShardDataset, make_batch_source
from parakeet.data.features import build_latent_cache, cache_teacher_corpus
from parakeet.data.text import TextTokenizer
from parakeet.models import build_model

from test_voice import VOICES, _multi_voice_manifest


def _corpus(fast_cfg, tmp_path, texts=("hello there", "another line"), mix=None):
    """A fixture corpus directory with a manifest and provenance, i.e. what make_teacher_corpus writes."""
    import soundfile as sf

    cfg = copy.deepcopy(fast_cfg)
    cfg.n_voices = 3
    manifest = _multi_voice_manifest(cfg, tmp_path, voices=VOICES, texts=texts)
    # spread the records across two teachers so per-sample mixture weights differ
    lines = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, rec in enumerate(lines):
        rec["teacher"] = "stub_low" if i % 2 == 0 else "stub_high"
    manifest.write_text(
        "\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8"
    )
    mix = mix or {"stub_low": 0.7, "stub_high": 0.3}
    (tmp_path / "corpus_meta.json").write_text(
        json.dumps({"mix": mix, "teachers": {}, "n_utterances": len(texts) * len(VOICES)}),
        encoding="utf-8",
    )
    return cfg, manifest, mix


# ------------------------------------------------------------------ batch source factory
def test_make_batch_source_pairs_references_only_for_the_flow_stage(fast_cfg, tmp_path):
    cfg, manifest, _mix = _corpus(fast_cfg, tmp_path)
    model = build_model(cfg)
    cache = build_latent_cache(
        manifest, tmp_path / "cache", cfg, model.autoencoder,
        tokenizer=TextTokenizer(mode=cfg.text.mode),
    )

    flow = make_batch_source(cfg, "flow", cache, batch_size=2, shuffle=False)
    batch = flow()
    assert "ref_mel" in batch and "ref_mel_neg" in batch, "flow must be paired by default"
    assert isinstance(flow, LatentShardBatchSource) and flow.pair_references

    text = make_batch_source(cfg, "distill-text", cache, batch_size=2, shuffle=False)
    assert not text.pair_references
    # the reference is still supplied (conditioning is not the flow stage's private business),
    # but there is no cross-sample negative pair
    assert "ref_mel" not in text() or "ref_mel_neg" not in text()

    override = make_batch_source(
        cfg, "flow", cache, batch_size=2, shuffle=False, pair_references=False
    )
    assert not override.pair_references


def test_make_batch_source_honours_max_ref_frames(fast_cfg, tmp_path):
    cfg, manifest, _mix = _corpus(fast_cfg, tmp_path)
    cfg.train.max_ref_frames = 24
    model = build_model(cfg)
    cache = build_latent_cache(
        manifest, tmp_path / "cache_n", cfg, model.autoencoder,
        tokenizer=TextTokenizer(mode=cfg.text.mode),
    )
    batch = make_batch_source(cfg, "flow", cache, batch_size=2, shuffle=False)()
    assert batch["ref_mel"].shape[-1] == 24, "the config cap must reach the batch"
    assert batch["ref_mask"].all(), "a truncated reference is fully valid, just shorter"


def test_make_batch_source_falls_back_to_synthetic_batches(fast_cfg):
    # no cache -> a dry run must still work (this is what `train.py --dry-run` does)
    batch = make_batch_source(fast_cfg, "distill-text", None, batch_size=2)()
    assert "ids" in batch and "latent_token" in batch


# ------------------------------------------------------------------ corpus -> cache helper
def test_cache_teacher_corpus_keeps_the_mixture_from_provenance(fast_cfg, tmp_path):
    """Regression: the CLI used to pass no mixture, silently making every weight 1.0."""
    cfg, manifest, mix = _corpus(fast_cfg, tmp_path)
    model = build_model(cfg)
    cache = cache_teacher_corpus(tmp_path, tmp_path / "cache", cfg, model.autoencoder)
    meta = json.loads((cache / "cache_meta.json").read_text(encoding="utf-8"))
    assert meta["teacher_weights"] == mix, "the mixture must come from corpus_meta.json"

    dataset = LatentShardDataset(cache)
    weights = {round(float(dataset[i]["teacher_weight"]), 6) for i in range(len(dataset))}
    assert len(weights) > 1, f"per-sample weights should reflect the mixture, got {weights}"


def test_cache_teacher_corpus_prefers_the_curated_manifest(fast_cfg, tmp_path):
    cfg, manifest, _mix = _corpus(fast_cfg, tmp_path)
    model = build_model(cfg)
    lines = [l for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    curated = tmp_path / "kept.jsonl"
    curated.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    cache = cache_teacher_corpus(tmp_path, tmp_path / "cache_curated", cfg, model.autoencoder)
    dataset = LatentShardDataset(cache)
    assert len(dataset) == 2, "curation must actually change what gets cached"


def test_cache_teacher_corpus_finds_a_nested_curated_manifest(fast_cfg, tmp_path):
    """The CLI writes <corpus>/curated/kept.jsonl; the helper must find it there too."""
    cfg, manifest, _mix = _corpus(fast_cfg, tmp_path)
    model = build_model(cfg)
    lines = [l for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    nested = tmp_path / "curated"
    nested.mkdir()
    (nested / "kept.jsonl").write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    cache = cache_teacher_corpus(tmp_path, tmp_path / "cache_nested", cfg, model.autoencoder)
    assert len(LatentShardDataset(cache)) == 2


def test_cache_teacher_corpus_requires_a_manifest(fast_cfg, tmp_path):
    cfg, _manifest, _mix = _corpus(fast_cfg, tmp_path / "corpus_ok")
    empty = tmp_path / "empty"
    empty.mkdir()
    model = build_model(cfg)
    with pytest.raises(FileNotFoundError, match="kept.jsonl"):
        cache_teacher_corpus(empty, tmp_path / "cache_x", cfg, model.autoencoder)


# ------------------------------------------------------------------ curation actually runs
def test_curation_gates_reject_silence_and_discriminate(fast_cfg, tmp_path):
    """`curate_manifest` was dead code; this pins that the gates discriminate.

    Note the honest scope: the *published* thresholds are calibrated for real 24 kHz speech
    (CosyVoice: segments >= 3 s; PilotTTS/CosyVoice bandwidth >= 5 kHz), and the fixtures are
    ~0.35 s band-limited synthetic stacks, so the published gates reject **all** of them.  That is a
    property of the fixture, not a defect in the gates -- so the assertion is on *discrimination*
    (silence must attract strictly more reasons than speech) plus a fixture-appropriate config where
    the speech survives.
    """
    import numpy as np
    import soundfile as sf

    cfg, manifest, _mix = _corpus(fast_cfg, tmp_path)
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sr = cfg.audio.sample_rate

    silence_rel = "wav/silence.wav"
    sf.write(str(tmp_path / silence_rel), np.zeros(sr, dtype=np.float32), sr)
    records.append(
        {
            "utt_id": "silence",
            "text": "hello there",
            "teacher": "stub_low",
            "voice": "v0",
            "wav_path": silence_rel,
            "sample_rate": sr,
            "duration_s": 1.0,
        }
    )

    def load_wav(rel: str):
        wav, rate = sf.read(str(tmp_path / rel), dtype="float32")
        return torch.from_numpy(wav), rate

    out = tmp_path / "curated"
    published = curate_manifest(records, load_wav, out, CurateConfig(), normalize=True)
    assert published.n_total == len(records)
    assert published.n_rejected >= 1
    silence_reasons = next(
        len(r["quality"]["reasons"])
        for r in _jsonl(out / "rejected.jsonl")
        if r["utt_id"] == "silence"
    )
    speech_reasons = max(
        (
            len(r["quality"]["reasons"])
            for r in _jsonl(out / "rejected.jsonl")
            if r["utt_id"] != "silence"
        ),
        default=0,
    )
    assert silence_reasons > speech_reasons, (
        f"silence must be rejected for more reasons than band-limited speech "
        f"({silence_reasons} vs {speech_reasons})"
    )
    assert (out / "curation_report.json").exists()

    # fixture-appropriate gates: the fixture speech must survive, and silence must **still** be
    # rejected.  Relaxing duration/bandwidth/SNR previously let digital silence through: the
    # silence_ratio rule is relative to the file's own peak, so a zero peak scored 0.0, and there
    # was no absolute level gate at all.
    relaxed = CurateConfig(min_duration_s=0.1, min_bandwidth_hz=0.0, min_snr_db=0.0)
    report = curate_manifest(records, load_wav, tmp_path / "curated_relaxed", relaxed, normalize=True)
    assert report.n_kept >= 1, report.reason_counts
    assert report.n_rejected >= 1, "silence must be rejected even with relaxed gates"
    kept = _jsonl(tmp_path / "curated_relaxed" / "kept.jsonl")
    rejected = _jsonl(tmp_path / "curated_relaxed" / "rejected.jsonl")
    assert all(r["utt_id"] != "silence" for r in kept)
    assert all(r["quality"]["keep"] for r in kept)
    bad = next(r for r in rejected if r["utt_id"] == "silence")
    assert any("low_level" in reason for reason in bad["quality"]["reasons"]), bad["quality"]["reasons"]
    assert bad["quality"]["silence_ratio"] == 1.0, "digital silence is entirely silence"
    assert any("low_level" in r for r in bad["quality"]["reasons"])


def _jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_curated_cache_round_trips_through_the_recipe(fast_cfg, tmp_path):
    """synthesize -> curate -> cache: the curated manifest must be a valid cache input."""
    cfg, manifest, _mix = _corpus(fast_cfg, tmp_path)
    model = build_model(cfg)
    import soundfile as sf

    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    def load_wav(rel: str):
        wav, rate = sf.read(str(tmp_path / rel), dtype="float32")
        return torch.from_numpy(wav), rate

    relaxed = CurateConfig(min_duration_s=0.1, min_bandwidth_hz=0.0, min_snr_db=0.0)
    curate_manifest(records, load_wav, tmp_path / "curated", relaxed, normalize=True)
    cache = cache_teacher_corpus(tmp_path, tmp_path / "cache_r", cfg, model.autoencoder)
    dataset = LatentShardDataset(cache)
    assert len(dataset) >= 1
    item = dataset[0]
    assert int(item["ids"].numel()) == int(item["f0"].numel())
    assert "voice" in item and "teacher_weight" in item
