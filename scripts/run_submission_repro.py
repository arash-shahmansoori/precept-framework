#!/usr/bin/env python3
"""
Submission reproducibility runner.

This script provides a one-command, reviewer-friendly reproducibility check for
the curated submission package in `submission_repro_data/`.

It performs three actions:
1) Validates environment lock checksums.
2) Validates paper-figure hash parity (paper file vs curated source file).
3) Reconstructs a paper-artifact bundle under an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION_DIR = ROOT / "submission_repro_data"
DEFAULT_OUTPUT_DIR = DEFAULT_SUBMISSION_DIR / "regenerated_paper_artifacts"


TABLE_FILES = [
    "submission_repro_data/publication_results/exp1_main_comparison_combined/tables/table1_main_comparison.tex",
    "submission_repro_data/publication_results/exp1_main_comparison_combined/tables/table1_main_comparison.md",
    "submission_repro_data/publication_results/exp1_main_comparison_combined/tables/statistical_summary.md",
    "submission_repro_data/publication_results/exp6_final_publication/exp6_table.tex",
    "submission_repro_data/publication_results/EXP5_RULE_PERSISTENCE_VERIFICATION_REPORT.md",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_figure_hash_file(path: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("PAPER"):
            if current:
                entries.append(current)
                current = {}
            current["paper"] = line.split(None, 1)[1].strip()
        elif line.startswith("SOURCE"):
            current["source"] = line.split(None, 1)[1].strip()
        elif line.startswith("HASH_PAPER"):
            current["hash_paper"] = line.split(None, 1)[1].strip()
        elif line.startswith("HASH_SOURCE"):
            current["hash_source"] = line.split(None, 1)[1].strip()
        elif line.startswith("MATCH"):
            current["match"] = line.split(None, 1)[1].strip()

    if current:
        entries.append(current)

    return entries


def validate_environment_lock(submission_dir: Path) -> Dict[str, object]:
    env_dir = submission_dir / "environment"
    lock_json_path = env_dir / "environment_lock.json"
    req_lock_path = env_dir / "requirements.lock.txt"
    uv_lock_path = ROOT / "uv.lock"

    if not lock_json_path.exists():
        raise FileNotFoundError(f"Missing environment lock metadata: {lock_json_path}")
    if not req_lock_path.exists():
        raise FileNotFoundError(f"Missing requirements lock file: {req_lock_path}")
    if not uv_lock_path.exists():
        raise FileNotFoundError(f"Missing uv lock file: {uv_lock_path}")

    lock_meta = json.loads(lock_json_path.read_text())

    uv_lock_actual = sha256_file(uv_lock_path)
    req_lock_actual = sha256_file(req_lock_path)

    uv_version = "unavailable"
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    except Exception:
        # Keep going; lock validation still works without installed uv
        pass

    result = {
        "uv_lock_expected": lock_meta.get("uv_lock_sha256"),
        "uv_lock_actual": uv_lock_actual,
        "uv_lock_match": uv_lock_actual == lock_meta.get("uv_lock_sha256"),
        "requirements_expected": lock_meta.get("requirements_lock_sha256"),
        "requirements_actual": req_lock_actual,
        "requirements_match": req_lock_actual == lock_meta.get("requirements_lock_sha256"),
        "uv_version_expected": lock_meta.get("uv_version_used_for_lock_export"),
        "uv_version_actual": uv_version,
        "uv_version_match": uv_version == lock_meta.get("uv_version_used_for_lock_export"),
    }
    return result


def validate_figure_hashes(submission_dir: Path) -> Dict[str, object]:
    hash_file = submission_dir / "FIGURE_SHA256.txt"
    if not hash_file.exists():
        raise FileNotFoundError(f"Missing figure hash file: {hash_file}")

    entries = parse_figure_hash_file(hash_file)
    if not entries:
        raise ValueError(f"No entries found in {hash_file}")

    checks = []
    all_ok = True
    for entry in entries:
        paper_path = ROOT / entry["paper"]
        source_path = ROOT / entry["source"]
        if not paper_path.exists() or not source_path.exists():
            checks.append(
                {
                    "paper": entry.get("paper"),
                    "source": entry.get("source"),
                    "ok": False,
                    "reason": "missing_file",
                }
            )
            all_ok = False
            continue

        paper_hash = sha256_file(paper_path)
        source_hash = sha256_file(source_path)
        ok = (
            paper_hash == source_hash
            and paper_hash == entry.get("hash_paper")
            and source_hash == entry.get("hash_source")
        )
        checks.append(
            {
                "paper": entry["paper"],
                "source": entry["source"],
                "paper_hash": paper_hash,
                "source_hash": source_hash,
                "ok": ok,
            }
        )
        all_ok = all_ok and ok

    return {"ok": all_ok, "count": len(checks), "checks": checks}


def reconstruct_bundle(submission_dir: Path, output_dir: Path) -> Dict[str, object]:
    hash_entries = parse_figure_hash_file(submission_dir / "FIGURE_SHA256.txt")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "paper_figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "docs").mkdir(parents=True, exist_ok=True)

    copied_figures = []
    copied_tables = []
    copied_docs = []

    # Copy canonical paper figures from curated source paths
    for entry in hash_entries:
        src = ROOT / entry["source"]
        dst = output_dir / "paper_figures" / src.name
        shutil.copy2(src, dst)
        copied_figures.append(str(dst.relative_to(ROOT)))

        # Also include same-name PDF if present in source directory
        pdf_src = src.with_suffix(".pdf")
        if pdf_src.exists():
            pdf_dst = output_dir / "paper_figures" / pdf_src.name
            shutil.copy2(pdf_src, pdf_dst)
            copied_figures.append(str(pdf_dst.relative_to(ROOT)))

    # Copy key table/summary files
    for rel_path in TABLE_FILES:
        src = ROOT / rel_path
        if src.exists():
            dst = output_dir / "paper_tables" / src.name
            shutil.copy2(src, dst)
            copied_tables.append(str(dst.relative_to(ROOT)))

    # Copy package docs
    doc_files = [
        submission_dir / "REPRODUCIBILITY_MANIFEST.md",
        submission_dir / "paper_experiment_sources" / "README.md",
        submission_dir / "FIGURE_SHA256.txt",
        submission_dir / "environment" / "ENVIRONMENT_LOCK.md",
        submission_dir / "environment" / "environment_lock.json",
    ]
    for src in doc_files:
        if src.exists():
            dst = output_dir / "docs" / src.name
            shutil.copy2(src, dst)
            copied_docs.append(str(dst.relative_to(ROOT)))

    return {
        "output_dir": str(output_dir.relative_to(ROOT)),
        "copied_figures": copied_figures,
        "copied_tables": copied_tables,
        "copied_docs": copied_docs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run submission reproducibility checks")
    parser.add_argument(
        "--submission-dir",
        default=str(DEFAULT_SUBMISSION_DIR),
        help="Path to submission reproducibility package",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Path to write reconstructed artifacts",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip environment lock checksum validation",
    )
    parser.add_argument(
        "--skip-figure-check",
        action="store_true",
        help="Skip paper/source figure hash validation",
    )
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not submission_dir.exists():
        print(f"[ERROR] Submission directory not found: {submission_dir}")
        return 1

    report: Dict[str, object] = {
        "submission_dir": str(submission_dir),
        "output_dir": str(output_dir),
    }

    try:
        if not args.skip_env_check:
            env_result = validate_environment_lock(submission_dir)
            report["environment_check"] = env_result
            if not (env_result["uv_lock_match"] and env_result["requirements_match"]):
                print("[ERROR] Environment lock checksum mismatch.")
                print(json.dumps(env_result, indent=2))
                return 2
            print("[OK] Environment lock checks passed.")
        else:
            report["environment_check"] = {"skipped": True}

        if not args.skip_figure_check:
            fig_result = validate_figure_hashes(submission_dir)
            report["figure_hash_check"] = fig_result
            if not fig_result["ok"]:
                print("[ERROR] Figure hash checks failed.")
                return 3
            print(f"[OK] Figure hash checks passed ({fig_result['count']} figures).")
        else:
            report["figure_hash_check"] = {"skipped": True}

        reconstruction = reconstruct_bundle(submission_dir, output_dir)
        report["reconstruction"] = reconstruction

        report_path = output_dir / "repro_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[OK] Reconstructed artifact bundle: {output_dir}")
        print(f"[OK] Report written: {report_path}")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
