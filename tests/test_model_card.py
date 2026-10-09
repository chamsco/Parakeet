"""The model card must be *derived* from evidence, and must fail when the evidence is not there.

A model card is usually prose that drifts from reality: numbers copied from an old run, a limitation
quietly dropped, a claim whose measurement stopped being produced.  This one is generated from the
report files the demos write, with the JSON path recorded for every row -- and the tests below are
about the *check* having teeth, not about the prose reading well:

* every advertised claim must reference a report that exists in the repository;
* a missing report must come back ``unmeasured`` (never silently dropped, never a pass);
* a failing measurement must come back ``fail``;
* the licence table must be generated from the teacher specs, not typed in.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_card import CLAIMS, build, evaluate, render  # noqa: E402


def test_every_claim_cites_committed_evidence():
    """The anti-drift guard: a claim whose evidence is not in the repository is unverifiable.

    The raw reports live in ``runs/`` (gitignored), so the card cites the committed snapshot in
    ``docs/evidence/``.  Both the snapshot and the run report it came from must be recorded, and the
    manifest's hashes must match the files on disk.
    """
    import hashlib

    manifest_path = ROOT / "docs" / "evidence" / "manifest.json"
    assert manifest_path.exists(), "run: python scripts/model_card.py --refresh-evidence"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {e["claim"]: e for e in manifest["entries"]}

    for claim in CLAIMS:
        assert claim["id"] in entries, f"{claim['id']} has no evidence entry"
        entry = entries[claim["id"]]
        assert entry["status"] == "copied", f"{claim['id']}: {entry}"
        evidence = ROOT / entry["evidence"]
        assert evidence.exists(), f"{claim['id']}: evidence file missing ({entry['evidence']})"
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{claim['id']}: evidence changed without refreshing the manifest"
        )
        assert claim["report"], f"{claim['id']} must also record where a fresh run writes it"


def test_every_claim_is_a_meaningful_assertion():
    for claim in CLAIMS:
        assert claim["statement"].strip(), claim["id"]
        assert claim["report"], claim["id"]
        assert claim["op"] in {">=", "<=", "==", "all_true", "lt_field", "sum<="}, claim["id"]
        # a threshold-less claim must be a comparison or an all-true check
        if claim["op"] in {">=", "<=", "=="}:
            assert "target" in claim, claim["id"]


def test_the_real_evidence_satisfies_every_claim():
    """If this fails, either a measurement regressed or a report stopped being produced."""
    results, summary = build()
    failing = [r for r in results if r["status"] == "fail"]
    unmeasured = [r for r in results if r["status"] == "unmeasured"]
    assert not unmeasured, f"unbacked claims: {[r['id'] for r in unmeasured]}"
    assert not failing, f"failing claims: {[(r['id'], r['detail']) for r in failing]}"
    assert summary["n_pass"] == len(CLAIMS) > 10


def test_a_missing_report_is_unmeasured_not_a_pass(tmp_path):
    """Positive control for the check itself."""
    results, summary = build(reports_root=tmp_path)  # an empty root: nothing exists
    assert summary["n_pass"] == 0
    assert summary["n_unmeasured"] == len(CLAIMS)
    assert all(r["status"] == "unmeasured" for r in results)
    assert all(r["report_used"] is None for r in results)
    assert all("no report" in r["detail"] for r in results)


def test_a_failing_measurement_is_reported_as_failing(tmp_path):
    """Positive control, the other direction: bad numbers must not be smoothed over."""
    claim = next(c for c in CLAIMS if c["op"] == "==")
    report = tmp_path / claim["report"][0]
    report.parent.mkdir(parents=True, exist_ok=True)
    # the real report's value, deliberately wrong
    payload = json.loads((ROOT / claim["report"][0]).read_text(encoding="utf-8"))
    payload[claim["field"]] = 0.5
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate(claim, reports_root=tmp_path)
    assert result["status"] == "fail", result
    assert result["value"] == 0.5

    # and a claim whose report exists but whose field is gone is unmeasured, not silently passing
    payload.pop(claim["field"])
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert evaluate(claim, reports_root=tmp_path)["status"] == "unmeasured"


def test_all_true_claim_reports_which_check_failed(tmp_path):
    claim = next(c for c in CLAIMS if c["op"] == "all_true")
    report = tmp_path / claim["report"][0]
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads((ROOT / claim["report"][0]).read_text(encoding="utf-8"))
    key = next(iter(payload[claim["field"]]))
    payload[claim["field"]][key] = False
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = evaluate(claim, reports_root=tmp_path)
    assert result["status"] == "fail"
    assert key in result["detail"], "the failing check must be named"


# ------------------------------------------------------------------ the rendered document
def test_rendered_card_states_the_limitations_it_cannot_measure_away():
    results, _summary = build()
    card = render(results, type("A", (), {"out": "x"})())
    lowered = card.lower()
    for required in (
        "no trained checkpoint",
        "out of scope and misuse",
        "voice cloning without the speaker's consent",
        "utmos",
        "intended use",
    ):
        assert required in lowered, f"the card must state: {required}"


def test_licence_table_comes_from_the_teacher_specs():
    from parakeet.data.teacher import TEACHERS

    results, _summary = build()
    card = render(results, type("A", (), {"out": "x"})())
    for name, spec in TEACHERS.items():
        assert f"`{name}`" in card, name
        assert spec.weights_license[:20] in card, f"{name} licence text missing"
    # the restricted teacher must be visibly restricted
    assert "**no**" in card, "a non-trainable teacher must be marked"
    assert "minimax" in card and "restricted" in card.lower()


def test_card_reports_the_parameter_counts_it_computes():
    results, _summary = build()
    card = render(results, type("A", (), {"out": "x"})())
    from parakeet.models import build_model, count_parameters
    from parakeet.config import load_config

    tiny = count_parameters(build_model(load_config("configs/parakeet_tiny.yaml")))
    assert f"{tiny / 1e6:.3f} M" in card, "the card must show the computed Tiny size"
