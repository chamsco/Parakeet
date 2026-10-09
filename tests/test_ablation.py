"""The capacity ablation and the lighter config it recommends.

"Make it as light as possible" is the objective, and the text side is the largest component of
Parakeet-Tiny.  ``scripts/ablate.py`` measured the frontier on fixtures; these tests keep the
measured recommendation and the offered config from drifting apart, and keep the lite config
actually working (a config that loads but cannot train or synthesize is worse than none).
"""

import json
from pathlib import Path

import pytest
import torch

from parakeet.config import load_config
from parakeet.data.dataset import SyntheticBatchSource
from parakeet.inference import Synthesizer
from parakeet.models import build_model, count_parameters
from parakeet.train.stages import run_stage

ROOT = Path(__file__).resolve().parents[1]
LITE = "configs/parakeet_tiny_lite.yaml"
SHIPPED = "configs/parakeet_tiny.yaml"


def test_lite_config_is_smaller_than_the_shipped_one():
    shipped = build_model(load_config(SHIPPED))
    lite = build_model(load_config(LITE))
    shipped_params = count_parameters(shipped)
    lite_params = count_parameters(lite)
    assert lite_params < shipped_params, "the lite variant must actually be lighter"
    shipped_text = sum(p.numel() for p in shipped.text.parameters())
    lite_text = sum(p.numel() for p in lite.text.parameters())
    assert lite_text < 0.5 * shipped_text, (
        f"the lite text side ({lite_text/1e6:.3f} M) should be well under half of the shipped one "
        f"({shipped_text/1e6:.3f} M)"
    )


def test_lite_config_trains_and_synthesizes():
    cfg = load_config(LITE)
    cfg.train.log_every = 10**6
    cfg.train.save_every = 0
    torch.manual_seed(0)
    model = build_model(cfg)
    source = SyntheticBatchSource(cfg, "distill-text", batch_size=2, n_frames=32, n_tokens=8, seed=0)
    logs = run_stage("distill-text", cfg, model=model, batches=source, max_steps=2,
                     out_dir="/tmp/parakeet_lite_check")
    assert logs["loss"] == logs["loss"], "a NaN loss means the config does not actually train"
    synth = Synthesizer(model, cfg, device="cpu", apply_phase_lock=True)
    wav = synth.synthesize("a light configuration", seed=0, steps=2)
    assert wav.shape[-1] > 0 and bool(torch.isfinite(wav).all())


def test_lite_config_matches_the_measured_recommendation():
    """The offered config must be the geometry the ablation actually recommended.

    Reads the committed evidence rather than a fresh run, so this holds in a clone (``runs/`` is
    gitignored) and fails if someone edits the config without re-measuring.
    """
    evidence = ROOT / "docs" / "evidence" / "capacity_ablation.json"
    if not evidence.exists():
        pytest.skip("no ablation evidence committed")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    recommended = payload["half_size_best"]
    # the summary carries the label; the geometry lives in the matching row
    row = next(r for r in payload["rows"] if r["label"] == recommended["label"])
    cfg = load_config(LITE)
    assert cfg.text.dim == row["text_dim"], (
        f"config text.dim={cfg.text.dim} but the ablation recommended {recommended['label']}"
    )
    assert cfg.text.n_layers == row["text_layers"]
    # and the recommendation must genuinely be lighter than the shipped configuration
    assert recommended["text_params"] < payload["shipped"]["text_params"]


def test_ablation_evidence_records_its_own_limitations():
    """A study without its caveat is an overclaim."""
    payload = json.loads((ROOT / "docs" / "evidence" / "capacity_ablation.json").read_text("utf-8"))
    caveat = payload.get("caveat", "")
    assert "synthetic" in caveat.lower() or "fixture" in caveat.lower()
    assert payload.get("split", {}).get("val", 0) >= 3, "a held-out split of 1-2 items is not a split"
    for row in payload["rows"]:
        assert "val_fit" in row and "train_fit" in row, "both splits must be reported"
