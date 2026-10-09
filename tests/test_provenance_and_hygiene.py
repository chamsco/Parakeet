"""Provenance, the documented freeze/checkpoint path, and repository hygiene.

Three findings from round 11, all of the same shape -- something declared but not actually working:

* ``write_run_metadata`` was **dead code**, so no run had ever recorded which code, config or data
  produced its weights;
* ``SpeakerConfig.checkpoint`` / ``SpeakerConfig.freeze`` were **never read**, so the documented
  production path (frozen CAM++ identity + an adapting Q-Former) did not exist, and every run
  trained the randomly-initialised stand-in;
* ``.gitignore``'s unanchored ``data/`` had kept the **entire ``parakeet/data/`` package** out of the
  repository for the first ten commits -- the published repo could not even import.
"""

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest
import torch

from parakeet.config import ParakeetConfig, load_config
from parakeet.data.dataset import SyntheticBatchSource
from parakeet.models import build_model
from parakeet.models.speaker import SpeakerConditioner
from parakeet.train.common import (
    config_fingerprint,
    count_trainable,
    file_fingerprint,
    git_revision,
    write_run_metadata,
)
from parakeet.train.stages import run_stage

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ repo hygiene
def _tracked() -> set[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except Exception:  # noqa: BLE001
        pytest.skip("git unavailable")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return {line.strip().replace("\\", "/") for line in out.stdout.splitlines() if line.strip()}


def test_every_package_source_file_is_tracked_by_git():
    """Regression: an unanchored `data/` ignore rule hid the whole data package from the repo.

    Ten commits were pushed in which `import parakeet.data` failed on a fresh clone while working
    perfectly in the working tree, because git silently ignored the directory.  Any source file in
    the package that git does not track is a file users do not get.
    """
    tracked = _tracked()
    on_disk = {
        p.relative_to(ROOT).as_posix() for p in (ROOT / "parakeet").rglob("*.py")
    }
    assert on_disk, "no sources found under parakeet/ -- wrong working directory?"
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "these package files are not tracked by git, so they are absent from the repository "
        f"(check .gitignore for an unanchored pattern): {missing}"
    )


def test_dev_side_files_are_not_accidentally_ignored():
    """The scripts and CI config must ship too (the corpus/runs ignore rules are root-anchored)."""
    tracked = _tracked()
    required = {
        "scripts/train.py",
        "scripts/recipe_dry_run.py",
        "scripts/make_teacher_corpus.py",
        ".github/workflows/ci.yml",
        "configs/parakeet_tiny.yaml",
        "configs/parakeet_small.yaml",
    }
    missing = sorted(required - tracked)
    assert not missing, f"expected these to be tracked: {missing}"


def test_no_dead_public_api_in_the_package():
    """A public name nothing references is a capability that does not exist.

    Round 11 found 17 (``write_run_metadata``, ``consistency_distillation_loss``, a dozen superseded
    math helpers).  Rather than rediscover them, this asserts the list stays empty.  If a name is
    genuinely part of the public API with no internal caller, add it to ``ALLOWED`` with a reason --
    which is the point: the decision becomes explicit instead of accidental.
    """
    allowed: dict[str, str] = {}

    pkg = ROOT / "parakeet"
    files = [
        p
        for d in (pkg, ROOT / "scripts", ROOT / "tests")
        for p in d.rglob("*.py")
    ]
    texts = {p: p.read_text(encoding="utf-8", errors="ignore") for p in files}

    def_lines: set[tuple[Path, int]] = set()
    names: dict[str, Path] = {}
    for path in sorted(pkg.rglob("*.py")):
        tree = ast.parse(texts[path])
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                names[node.name] = path
                def_lines.add((path, node.lineno))

    dead = []
    for name, own in names.items():
        if name in allowed:
            continue
        hits = 0
        for path, text in texts.items():
            for i, line in enumerate(text.splitlines(), start=1):
                if (path, i) in def_lines:
                    continue
                if re.search(rf"\b{re.escape(name)}\b", line):
                    hits += 1
        if hits == 0:
            dead.append(f"{name} ({own.relative_to(ROOT).as_posix()})")
    assert not dead, (
        "public names referenced nowhere: " + ", ".join(sorted(dead)) + ". Wire them, delete them, "
        "or add them to ALLOWED with a reason."
    )


# ------------------------------------------------------------------ provenance
def test_run_stage_writes_provenance(fast_cfg, tmp_path):
    cfg = ParakeetConfig(variant="tiny")
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=32, n_tokens=8)
    run_stage("distill-text", cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path))

    meta = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert meta["stage"] == "distill-text"
    assert meta["config_sha256"] == config_fingerprint(cfg)
    assert meta["environment"]["python"] and meta["environment"]["torch"]
    assert meta["extra"]["trainable_params"] > 0
    assert meta["extra"]["total_params"] >= meta["extra"]["trainable_params"]
    # the frozen/trainable report that would have caught "this stage trained nothing"
    assert "autoencoder" in meta["extra"]["frozen_modules"], meta["extra"]["frozen_modules"]
    assert "text" not in meta["extra"]["frozen_modules"]

    # git provenance is best effort: present in a checkout, absent in a wheel
    git = git_revision()
    assert (meta["git"] is None) == (git is None)
    if git:
        assert meta["git"]["rev"] and isinstance(meta["git"]["dirty"], bool)


def test_run_stage_records_caller_metadata(fast_cfg, tmp_path):
    cfg = ParakeetConfig(variant="tiny")
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=32, n_tokens=8)
    run_stage(
        "distill-text", cfg, model=model, batches=source, max_steps=1, out_dir=str(tmp_path),
        run_metadata={"teachers": ["orpheus", "kokoro"], "teacher_weights": {"orpheus": 0.6}},
    )
    meta = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert meta["extra"]["teachers"] == ["orpheus", "kokoro"]
    assert meta["extra"]["teacher_weights"]["orpheus"] == 0.6


def test_config_fingerprint_changes_with_the_recipe(fast_cfg):
    a = ParakeetConfig(variant="tiny")
    b = ParakeetConfig(variant="tiny")
    assert config_fingerprint(a) == config_fingerprint(b), "same config, same hash"
    b.train.lr = a.train.lr * 2
    assert config_fingerprint(a) != config_fingerprint(b), "a changed recipe must change the hash"


def test_file_fingerprint_is_content_addressed(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    first = file_fingerprint(path)
    assert first and len(first) == 64
    path.write_text('{"a": 1}\n', encoding="utf-8")
    assert file_fingerprint(path) == first, "same bytes, same hash"
    path.write_text('{"a": 2}\n', encoding="utf-8")
    assert file_fingerprint(path) != first
    assert file_fingerprint(tmp_path / "absent.jsonl") is None


def test_count_trainable_tracks_freezing(fast_cfg):
    cfg = ParakeetConfig(variant="tiny")
    model = build_model(cfg)
    total = count_trainable(model)
    for p in model.autoencoder.parameters():
        p.requires_grad = False
    frozen_total = count_trainable(model)
    assert 0 < frozen_total < total
    assert total - frozen_total == sum(p.numel() for p in model.autoencoder.parameters())


def test_write_run_metadata_creates_its_directory(fast_cfg, tmp_path):
    target = tmp_path / "nested" / "run"
    path = write_run_metadata(target, ParakeetConfig(variant="tiny"), stage="autoencoder")
    assert path.exists() and path.parent == target
    assert json.loads(path.read_text(encoding="utf-8"))["stage"] == "autoencoder"


# ------------------------------------------------------------------ speaker freeze / checkpoint
def _speaker_cfg(tmp_path, checkpoint=None, freeze=True, n_voices=1):
    cfg = load_config("configs/parakeet_small.yaml")
    cfg.speaker.channels = [16, 24]
    cfg.speaker.emb_dim = 32
    cfg.speaker.style_dim = 32
    cfg.speaker.n_query = 4
    cfg.speaker.checkpoint = str(checkpoint) if checkpoint else None
    cfg.speaker.freeze = freeze
    cfg.n_voices = n_voices
    return cfg.validate()


def test_speaker_freeze_is_honoured(tmp_path):
    """`SpeakerConfig.freeze` was declared and never read: identity was always trainable."""
    cfg = _speaker_cfg(tmp_path, checkpoint=None, freeze=True)
    model = build_model(cfg)
    assert not any(p.requires_grad for p in model.speaker.speaker.parameters()), "must be frozen"
    # the Q-Former still adapts: that is the point of freezing only the identity encoder
    assert any(p.requires_grad for p in model.speaker.qformer.parameters())

    unfrozen = build_model(_speaker_cfg(tmp_path, checkpoint=None, freeze=False))
    assert all(p.requires_grad for p in unfrozen.speaker.speaker.parameters())


def test_speaker_checkpoint_is_loaded_from_a_full_model_state_dict(tmp_path):
    """The documented production path: load real CAM++ weights instead of the random stand-in."""
    source_cfg = _speaker_cfg(tmp_path, checkpoint=None, freeze=False)
    source = build_model(source_cfg)
    # re-randomise so the loaded weights are distinguishable from a fresh initialisation
    with torch.no_grad():
        for p in source.speaker.speaker.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ckpt = tmp_path / "campplus.pt"
    torch.save({"model": source.state_dict()}, ckpt)

    cfg = _speaker_cfg(tmp_path, checkpoint=ckpt, freeze=True)
    model = build_model(cfg)
    report = model.speaker.load_report
    assert report["loaded"] > 0, report
    assert report["frozen"] is True
    assert not any(p.requires_grad for p in model.speaker.speaker.parameters())

    reference = dict(source.speaker.speaker.state_dict())
    for key, value in model.speaker.speaker.state_dict().items():
        assert torch.allclose(value, reference[key]), f"{key} was not loaded from the checkpoint"


def test_speaker_checkpoint_accepts_a_raw_encoder_state_dict(tmp_path):
    source = build_model(_speaker_cfg(tmp_path, checkpoint=None, freeze=False))
    ckpt = tmp_path / "encoder_only.pt"
    torch.save(source.speaker.speaker.state_dict(), ckpt)
    model = build_model(_speaker_cfg(tmp_path, checkpoint=ckpt, freeze=False))
    assert model.speaker.load_report["loaded"] == len(source.speaker.speaker.state_dict())


def test_speaker_checkpoint_with_no_matching_tensors_raises(tmp_path):
    ckpt = tmp_path / "wrong.pt"
    torch.save({"model": {"totally.unrelated": torch.zeros(3)}}, ckpt)
    with pytest.raises(ValueError, match="expected tensors"):
        build_model(_speaker_cfg(tmp_path, checkpoint=ckpt))


def test_tiny_voice_mode_has_no_speaker_encoder(fast_cfg):
    """The Tiny variant conditions on a learned constant, so freezing must not break it."""
    cfg = ParakeetConfig(variant="tiny")
    cfg.speaker.checkpoint = None
    cfg.speaker.freeze = True
    model = build_model(cfg.validate())
    ids = torch.randint(1, 40, (2, 6))
    with torch.no_grad():
        out = model.text_side(ids)
    assert out["f0"].shape == (2, 6)
