"""Text/tokenisation, the teacher registry and its licence gate, and the corpus/cache path."""

import copy
import json

import numpy as np
import pytest
import torch

from parakeet.data.dataset import LatentShardDataset, SyntheticBatchSource, collate
from parakeet.data.features import LatentShardWriter, build_latent_cache, extract_signals
from parakeet.data.teacher import (
    DEFAULT_MIX,
    KOKORO,
    MINIMAX,
    ORPHEUS,
    TEACHERS,
    TeacherBackend,
    TeacherLicenseError,
    build_backend,
    check_teacher,
    resolve_mix,
    synthesize_corpus,
)
from parakeet.data.text import TAGS, CharVocab, TextTokenizer, normalize_text
from parakeet.models import build_model


# --------------------------------------------------------------------------- text
def test_normalize_text_numbers_and_case():
    assert normalize_text("HELLO 42 world!") == "hello four two world!"


def test_normalize_text_preserves_tags():
    out = normalize_text("Well <laugh> that's funny.")
    assert "<laugh>" in out
    assert out.startswith("well")


def test_tag_vocabulary_matches_teacher_controls():
    # every Orpheus/MiniMax tag we claim support for must be in the shared vocabulary
    for tag in ORPHEUS.supports_tags:
        assert tag in TAGS, tag
    tok = TextTokenizer()
    for tag in MINIMAX.supports_tags:
        assert tag in tok.vocab.tag_ids, tag


def test_tokenizer_roundtrip_and_batch():
    tok = TextTokenizer()
    ids, mask = tok.batch(["hello", "a longer sentence"], max_len=32)
    width = max(len(tok.encode("hello")), len(tok.encode("a longer sentence")))
    assert ids.shape == (2, width)
    assert mask.dtype == torch.bool
    assert mask[0].sum().item() == len(tok.encode("hello"))
    assert "hello" in tok.vocab.decode(ids[0].tolist())


def test_tokenizer_tags_are_separated_from_spoken_text():
    tok = TextTokenizer()
    clean, tags = tok.vocab.split_tags("I am <laugh> so happy <sigh>")
    assert tags == ["<laugh>", "<sigh>"]
    assert "<laugh>" not in clean and "<sigh>" not in clean
    assert tok.style_tags("hi <cough> there") == ["<cough>"]


def test_vocab_fits_model_capacity():
    for mode in ("char", "phoneme"):
        tok = TextTokenizer(mode=mode)
        assert tok.vocab_size <= 128, f"{mode} vocab {tok.vocab_size} exceeds the default embedding"
        assert tok.vocab_size > 40


def test_unknown_characters_become_unk():
    tok = TextTokenizer()
    ids = tok.encode("héllo ☃")
    assert tok.vocab.stoi["<unk>"] in ids


# --------------------------------------------------------------------------- teachers
def test_teacher_registry_licences():
    assert ORPHEUS.allows_training and KOKORO.allows_training
    assert not MINIMAX.allows_training and MINIMAX.restricted
    assert set(DEFAULT_MIX) == {"orpheus", "kokoro"}, "the default mixture must stay permissive"


def test_minimax_is_refused_by_default():
    with pytest.raises(TeacherLicenseError):
        check_teacher("minimax")
    spec = check_teacher("minimax", acknowledge_restricted=True)
    assert spec.restricted


def test_unknown_teacher_raises():
    with pytest.raises(KeyError):
        check_teacher("does-not-exist")


def test_build_backend_respects_gate():
    with pytest.raises(TeacherLicenseError):
        build_backend("minimax")


def test_resolve_mix_parsing():
    assert resolve_mix(None) == DEFAULT_MIX
    assert resolve_mix(["orpheus=0.7", "kokoro=0.3"]) == {"orpheus": 0.7, "kokoro": 0.3}
    assert resolve_mix(["orpheus"]) == {"orpheus": 1.0}


class DummyBackend(TeacherBackend):
    spec = KOKORO

    def __init__(self, sr: int = 24000) -> None:
        self.sr = sr

    def synthesize(self, text, voice=None):
        n = self.sr // 2
        t = np.arange(n, dtype=np.float32) / self.sr
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), self.sr


def test_synthesize_corpus_writes_manifest_and_mixture(tmp_path):
    texts = [f"sentence number {i}" for i in range(6)]
    manifest = synthesize_corpus(
        texts,
        tmp_path,
        mix={"kokoro": 1.0},
        backends={"kokoro": DummyBackend()},
        voices={"kokoro": ["af_heart"]},
    )
    assert manifest.exists()
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(records) == 6
    assert all(r["teacher"] == "kokoro" for r in records)
    assert all((tmp_path / r["wav_path"]).exists() for r in records)
    meta = json.loads((tmp_path / "corpus_meta.json").read_text(encoding="utf-8"))
    assert meta["n_utterances"] == 6
    assert meta["hours"] > 0


def test_synthesize_corpus_interleaves_mixture(tmp_path):
    texts = [f"text {i}" for i in range(20)]
    manifest = synthesize_corpus(
        texts,
        tmp_path,
        mix={"kokoro": 0.5, "orpheus": 0.5},
        backends={"kokoro": DummyBackend(), "orpheus": DummyBackend()},
    )
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    teachers = {r["teacher"] for r in records}
    assert teachers == {"kokoro", "orpheus"}, "both teachers must appear (no shard-local starvation)"


# --------------------------------------------------------------------------- features / cache
def test_extract_signals_shapes(fast_cfg):
    wav = 0.2 * torch.randn(1, 24000)
    ids = torch.arange(1, 11)
    sig = extract_signals(wav, fast_cfg, ids)
    assert sig.n_tokens == 10
    assert len(sig.durations) == 10
    assert len(sig.f0_bins) == sig.n_frames
    assert len(sig.energy_db) == sig.n_frames


def test_shard_writer_creates_index(tmp_path):
    writer = LatentShardWriter(tmp_path, shard_size=2)
    for i in range(5):
        writer.add({"latent": torch.randn(24, 10), "ids": torch.arange(4)})
    writer.flush()
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 3
    assert sum(e["n"] for e in index) == 5


def test_build_latent_cache_end_to_end(fast_cfg, tmp_path):
    import soundfile as sf

    corpus = tmp_path / "corpus"
    (corpus / "wav").mkdir(parents=True)
    lines = []
    for i in range(3):
        sr = fast_cfg.audio.sample_rate
        wav = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr, dtype=np.float32) / sr)).astype(np.float32)
        sf.write(str(corpus / "wav" / f"u{i}.wav"), wav, sr)
        lines.append(
            json.dumps(
                {
                    "utt_id": f"u{i}",
                    "text": f"hello there {i}",
                    "teacher": "kokoro",
                    "voice": "af_heart",
                    "wav_path": f"wav/u{i}.wav",
                    "sample_rate": sr,
                    "duration_s": 1.0,
                }
            )
        )
    (corpus / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    model = build_model(fast_cfg)
    out = build_latent_cache(corpus / "manifest.jsonl", tmp_path / "cache", fast_cfg, model.autoencoder)
    assert (out / "index.json").exists()
    assert (out / "cache_meta.json").exists()

    ds = LatentShardDataset(out)
    assert len(ds) == 3
    item = ds[0]
    assert item["latent"].shape[0] == fast_cfg.autoencoder.latent_dim
    batch = collate([ds[0], ds[1]])
    assert batch["ids"].shape[0] == 2
    assert batch["text_mask"].dtype == torch.bool
    assert batch["latent"].shape[1] == fast_cfg.autoencoder.latent_dim
    assert batch["durations"].shape[0] == 2


def test_synthetic_batch_source_keys(fast_cfg):
    for stage in ("autoencoder", "distill-decoder"):
        b = SyntheticBatchSource(fast_cfg, stage)()
        assert set(b) == {"wav"}
    b = SyntheticBatchSource(fast_cfg, "flow")()
    assert {"ids", "text_mask", "latent", "ref_mel", "ref_mask"} <= set(b)
    b = SyntheticBatchSource(fast_cfg, "distill-text")()
    assert {"durations", "f0", "energy", "latent_token"} <= set(b)
    with pytest.raises(ValueError):
        SyntheticBatchSource(fast_cfg, "bogus")()
