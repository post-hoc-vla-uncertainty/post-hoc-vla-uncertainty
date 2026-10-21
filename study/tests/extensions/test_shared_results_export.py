"""Publication preserves evidence, stays portable and fails without losing exports."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "shared_export", Path(__file__).resolve().parents[2] / "scripts/export_shared_results.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    blob = path.read_bytes()
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


@pytest.fixture
def project(tmp_path):
    for report, study in (
        ("reports/head_study", "storage/extensions/head_ablation_v1"),
        ("storage/extensions/followup_v1/report", "storage/extensions/followup_v1"),
    ):
        source = tmp_path / study
        refs = {}
        for name, value in {
            "round_000/head/result.json": dict(status="complete", metrics=dict(global_nauc=.4,
                gp_diagnostics=dict(kernels=[dict(loss_history=[3, 2, 1])]), mapping_check=dict(gp_r2=[.8]))),
            "round_000/head/scores.json": dict(scores={"condition_1": .25}),
            "declaration.json": dict(seeds=[0]),
            "queue.json": dict(running=["head"]),
            "diagnostics.json": dict(loss_history=list(range(100))),
        }.items():
            path = source / name
            refs[str(path)] = write(path, value)
        # A changing scheduler and heavy caches must not invalidate or enter an export.
        write(source / "queue.json", dict(running=["another_head"]))
        (source / "checkpoint.safetensors").write_bytes(b"weights")
        (source / "features.npy").write_bytes(b"embeddings")
        snapshot = tmp_path / report / "snapshots/test"
        write(snapshot / "input_hashes.json", refs)
        (snapshot / "report.html").write_text('<a href="metrics.csv">Download</a><script>window.ok=true;</script>')
        (snapshot / "metrics.csv").write_text("metric,value\nnauc,0.4\n")
        write(tmp_path / report / "latest.json", dict(snapshot="test", meta=dict(successful=1, total=2)))
    write(tmp_path / "storage/outputs/pretrain/seed0/checkpoints/last/train_config.json", dict(seed=0))
    return tmp_path


def test_portable_export_preserves_scores_and_omits_heavy_artifacts(project):
    output = export.build(project)
    assert export.verify(output) > 0
    result = json.loads((output / "original_study/evidence/round_000/head/result.json").read_text())
    assert result["metrics"] == dict(global_nauc=.4)
    manifest = json.loads((output / "manifest.json").read_text())
    evidence = output / "original_study/evidence/round_000/head/scores.json"
    assert evidence.read_bytes() == (project / "storage/extensions/head_ablation_v1/round_000/head/scores.json").read_bytes()
    assert all(p.suffix not in {".npz", ".npy", ".safetensors", ".pt"} for p in output.rglob("*"))
    assert not list(output.rglob("diagnostics.json")) and not list(output.rglob("queue.json"))
    assert (output / "reproduction/evidence/pretrain/seed0/checkpoint_metadata/last/train_config.json").exists()
    assert any(r["transformation"] != "none" for r in manifest["files"].values() if "transformation" in r)
    before = evidence.read_bytes()
    export.build(project)
    assert evidence.read_bytes() == before and not (project / ".shared_results-previous").exists()


def test_changed_source_keeps_previous_publication(project):
    output = export.build(project)
    before = (output / "manifest.json").read_bytes()
    write(project / "storage/extensions/head_ablation_v1/round_000/head/scores.json", dict(scores={"condition_1": 99}))
    with pytest.raises(ValueError, match="Source changed"):
        export.build(project)
    assert (output / "manifest.json").read_bytes() == before
    assert not list(project.glob(".shared_results-building-*"))
    export.verify(output)


def test_edited_bundle_is_not_silently_replaced(project):
    output = export.build(project)
    (output / "README.md").write_text("A collaborator's notes")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        export.build(project)
    assert (output / "README.md").read_text() == "A collaborator's notes"


def test_external_html_link_prevents_publication(project):
    page = project / "reports/head_study/snapshots/test/report.html"
    page.write_text('<a href="../../../../storage/checkpoint.bin">External artifact</a>')
    with pytest.raises(ValueError, match="Nonportable or missing link"):
        export.build(project)
    assert not (project / "shared_results").exists()
