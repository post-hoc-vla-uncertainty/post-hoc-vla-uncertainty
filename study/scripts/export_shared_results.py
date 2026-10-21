#!/usr/bin/env python3
"""Publish portable reports and compact evidence; never export model/cache files."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
OWNER = "vla-uncertainty/shared-results-v1"
REPORT_TYPES = {".html", ".csv", ".json", ".png", ".pdf"}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def compact_result(blob):
    """Keep scalar metrics and provenance; fit summaries remain in report exports."""
    value = json.loads(blob)
    metrics = value.get("metrics")
    if metrics:
        metrics.pop("gp_diagnostics", None)
        metrics.pop("mapping_check", None)
    return encode(value)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        key = "href" if tag in {"a", "link"} else "src" if tag in {"img", "script"} else None
        value = attrs.get(key, "")
        url = urlsplit(value)
        if value and not url.scheme and not url.netloc and url.path:
            self.local.append(unquote(url.path))


def validate_links(directory):
    directory = Path(directory).resolve()
    for page in directory.rglob("*.html"):
        parser = Links()
        parser.feed(page.read_text())
        for link in parser.local:
            target = (page.parent / link).resolve()
            if not target.is_relative_to(directory) or not target.is_file():
                raise ValueError(f"Nonportable or missing link in {page.name}: {link}")


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != OWNER:
        raise ValueError("Unrecognized shared-results manifest")
    files = manifest["files"]
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Bundle file list differs from its manifest")
    for name, record in files.items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Unexpected external file: {name}")
        blob = path.read_bytes()
        if len(blob) != record["bytes"] or digest(blob) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    validate_links(directory)
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


class Bundle:
    def __init__(self, root, destination):
        self.root, self.destination = Path(root).resolve(), Path(destination)
        self.files = {}
        self.omitted = []

    def put(self, relative, blob, **metadata):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in REPORT_TYPES | {".md"}:
            raise ValueError(f"Disallowed export path: {relative}")
        if len(blob) > 100_000_000:
            raise ValueError(f"Export file exceeds the 100 MB bundle guard: {relative}")
        key = str(relative)
        record = dict(bytes=len(blob), sha256=digest(blob), **metadata)
        if key in self.files and self.files[key] != record:
            raise ValueError(f"Conflicting export: {relative}")
        path = self.destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        self.files[key] = record

    def source(self, path, relative, *, expected=None, compact=False):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError(f"Source outside project or symlink: {path}")
        blob = path.read_bytes()
        source_hash = digest(blob)
        if expected and (source_hash != expected["sha256"] or len(blob) != expected["bytes"]):
            raise ValueError(f"Source changed after report capture; regenerate reports: {path}")
        self.put(relative, compact_result(blob) if compact else blob,
                 source=str(path.relative_to(self.root)), source_sha256=source_hash,
                 transformation="omit metrics.gp_diagnostics and metrics.mapping_check" if compact else "none")


def export_study(bundle, label, report_root, study_root):
    latest = json.loads((report_root / "latest.json").read_text())
    snapshot = report_root / "snapshots" / latest["snapshot"]
    if not snapshot.resolve().is_relative_to((report_root / "snapshots").resolve()):
        raise ValueError("Invalid report snapshot path")
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            if path.suffix not in REPORT_TYPES:
                raise ValueError(f"Unexpected report artifact: {path}")
            bundle.source(path, Path(label) / "report" / path.relative_to(snapshot))
    refs = json.loads((snapshot / "input_hashes.json").read_text())
    allowed = {"declaration.json", "training_data.json", "request.json", "result.json",
               "scores.json", "evaluation.json", "conditions.json"}
    captured = set()
    for name, expected in sorted(refs.items()):
        path = Path(name)
        if path.name not in allowed:
            bundle.omitted.append(dict(source=name, reason="Full fit diagnostics, bootstrap samples or live queue; summaries are in the report"))
            continue
        if path.is_relative_to(study_root):
            relative = path.relative_to(study_root)
        else:
            supervisor = bundle.root / "storage/extensions/supervisor/evaluation"
            relative = Path("supervisor_evaluation") / path.relative_to(supervisor)
        bundle.source(path, Path(label) / "evidence" / relative, expected=expected,
                      compact=label == "original_study" and path.name == "result.json")
        captured.add(path)
    if label == "followup_study":
        # These records are finalized before head evaluation. Match the report's
        # policies, not newly appearing heads from the still-running queue.
        policies = {p.parent.name for p in captured if p.name == "conditions.json"}
        extras = []
        for pid in sorted(policies):
            extras.extend((study_root / "evaluations" / pid).glob("rep_*/evaluation.json"))
            extras.extend((study_root / "policies" / pid).glob("model.json"))
        for name in ("configuration_matrix.json", "evaluation_manifest.json", "policy_quality.json",
                     "quality_gate.json", "observation_pairing.json", "source_migration.json"):
            if (study_root / name).exists():
                extras.append(study_root / name)
        for path in sorted(extras):
            bundle.source(path, Path(label) / "evidence" / path.relative_to(study_root))
    return dict(snapshot=latest["snapshot"], meta=latest["meta"])


def readme(studies, reproduction):
    a, b = (studies[key]["meta"] for key in ("original_study", "followup_study"))
    return f"""# Shared uncertainty-study results

Open the downloaded HTML files in a web browser for sortable tables, plots and
the interactive head inspector. GitHub's file viewer does not execute the HTML.
All plot/CSV download links are relative; keep each `report/` directory together.

| Study | Report | Completed evaluations |
| --- | --- | ---: |
| Original head study | [Open report](original_study/report/report.html) | {a['successful']}/{a['total']} ({a.get('eligible', a['successful'])} eligible) |
| Follow-up study | [Open report](followup_study/report/report.html) | {b['successful']}/{b['total']} |

These are frozen partial-study results, not claims that all planned experiments
finished. Exact capture times, statuses, protocol and source identities are in
`manifest.json` and each report's `snapshot.json`.

## What is included

- `original_study/report/` and `followup_study/report/`: complete published
  reports, figures and CSV exports, including paper baselines, matched ablations,
  confidence intervals and fit-milestone summaries.
- `*/evidence/`: head configurations and requests, individual uncertainty scores,
  compact results, paired learner/supervisor outcomes and study declarations.
  The follow-up also includes individual execution outcomes, policy-quality
  checks, training-data identities and evaluation-condition metadata.
- `reproduction/evidence/`: available original-reproduction numerical results,
  acquisition/training manifests and pretraining configuration/identity metadata.
  Checkpoint directories are renamed `checkpoint_metadata` and contain JSON only.
  Available round evaluations: {json.dumps(reproduction, ensure_ascii=False)}.
- `manifest.json`: SHA-256 and byte count for every published file, source hashes,
  export transformations, report snapshot IDs and Git revision of the exporter.

## How to compare heads

Start with `configuration_summary.csv`, `matched_pairs.csv` and
`paired_comparisons.csv` in each report directory. `episode_scores.csv` pairs
uncertainty with learner/supervisor outcomes; `task_scores.csv` contains the
task-level quantities used for the paper's correlation metrics.
`referral_curves.csv` includes random-referral and hindsight-oracle references.

The original study's seeds 42/43 change selected feature coordinates; they are
not independent policy-training seeds. The follow-up uses policy seeds 0/1/2
with paired supervisors and two training datasets (original and +35 demos).
Random feature subsets are averaged within a policy before averaging policies.
Each follow-up condition's outcome averages three executions. Treat the initial
condition as the resampling unit, retaining task and policy pairing; do not
count its three executions as three independent evaluation conditions.
The two studies have different protocols: compare methods within each study.

## Scope and portability

No policy/GP weights, datasets, observation images, embedding/projection caches,
bootstrap arrays, complete optimizer-loss traces, logs or historical preview
reports are included. The original compact `result.json` files omit full GP
diagnostics and mapping-check arrays; the report retains diagnostic summaries
and training milestones. Follow-up full `diagnostics.json` files are omitted.
All uncertainty scores and evaluation outcomes are copied without modification.

This bundle supports reviewing and independently recomputing statistical
comparisons without GPU inference. It is not a resumable experiment directory
or a drop-in input for the existing report generator, which also expects local
diagnostics/bootstrap artifacts. Producer-local paths inside provenance records
are informational; the report's links work without those paths. Source request
identities refer to the original artifacts; transformed file hashes are listed
separately in the bundle manifest. No historical failure is silently relabeled.

## Refresh and verify

On the experiment machine, regenerate both reports and replace this generated
bundle with one command (CPU only; running experiments are left alone):

```bash
python3 scripts/export_shared_results.py --refresh-reports
```

To export already-generated snapshots, omit `--refresh-reports`. Only results
captured in those snapshots are included, even if more heads finish during export.
To verify the downloaded bundle, using only the Python standard library:

```bash
python3 scripts/export_shared_results.py --verify
```

The exporter replaces its own generated directory on refresh. Edit the exporter
to change this README; keep collaborator annotations elsewhere in the repository.
"""


def build(root=ROOT):
    root = Path(root).resolve()
    output = root / "shared_results"
    if output.exists():
        # Avoid overwriting a user-created directory or edits to an earlier export.
        verify(output)
    temporary = Path(tempfile.mkdtemp(prefix=".shared_results-building-", dir=root))
    backup = root / ".shared_results-previous"
    if backup.exists():
        shutil.rmtree(temporary)
        raise ValueError("Previous publication backup exists; inspect it before retrying")
    try:
        bundle = Bundle(root, temporary)
        studies = {}
        for label, report, study in (
            ("original_study", "reports/head_study", "storage/extensions/head_ablation_v1"),
            ("followup_study", "storage/extensions/followup_v1/report", "storage/extensions/followup_v1"),
        ):
            print(f"[share] Exporting {label}", flush=True)
            studies[label] = export_study(bundle, label, root / report, root / study)
        reproduction = {}
        for category in ("reproduction", "pretrain"):
            source = root / "storage/outputs" / category
            for path in sorted(source.rglob("*.json")):
                if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
                    continue
                relative = Path(*(p if p != "checkpoints" else "checkpoint_metadata" for p in path.relative_to(source).parts))
                bundle.source(path, Path("reproduction/evidence") / category / relative)
                if path.name == "round_evaluation.json":
                    key = str(path.parent.parent.relative_to(source))
                    reproduction.setdefault(key, []).append(path.parent.name)
        bundle.put("README.md", readme(studies, reproduction).encode())
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
        manifest = dict(format=OWNER, exported_at=datetime.now(timezone.utc).isoformat(),
                        git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                        exporter_sha256=digest(Path(__file__).read_bytes()), studies=studies,
                        reproduction_evaluated_rounds=reproduction, files=bundle.files,
                        omitted_report_inputs=bundle.omitted)
        (temporary / "manifest.json").write_bytes(encode(manifest))
        total = verify(temporary)
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(temporary, output)
        except BaseException:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        print(f"[share] Published {output}: {total / 1e6:.1f} MB, {len(bundle.files) + 1} files; checksums and HTML links verified", flush=True)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh-reports", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        total = verify(ROOT / "shared_results")
        print(f"[share] Verified {total / 1e6:.1f} MB; checksums and HTML links intact")
        return
    if args.refresh_reports:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg")
        for command in (["bash", "scripts/experiments/09_head_report.sh"],
                        ["bash", "scripts/experiments/10_followup_study.sh", "--report-only"]):
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    build()


if __name__ == "__main__":
    main()
