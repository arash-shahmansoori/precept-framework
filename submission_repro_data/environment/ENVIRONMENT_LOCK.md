# Environment Lock

This reproducibility package uses a pinned dependency lock based on `uv.lock`.

## Locked Inputs

- `uv.lock` SHA-256:
  - `89292595870cd46f4c90d69d28d8991a5ada1a6b24298ae81e1eab0107db4f48`
- Exported pinned requirements:
  - `submission_repro_data/environment/requirements.lock.txt`
- Exported requirements SHA-256:
  - `fb41c6a3847b0b9988f89828c0b7f6eaa11697e5afa3e9a48922d286f633be96`
- `uv` version used when exporting:
  - `uv 0.6.2 (6d3614eec 2025-02-19)`

## One-Command Repro Entry Point

From project root:

`bash scripts/run_submission_repro.sh`

This command:

1. Syncs the environment from the lock file (`uv sync --frozen`).
2. Runs the reproducibility runner (`uv run scripts/run_submission_repro.py`).
3. Verifies figure hash parity and writes a reconstruction bundle.

## Manual Commands

If you prefer manual steps:

1. `uv sync --frozen`
2. `uv run scripts/run_submission_repro.py`

The runner verifies lock checksums and writes output to:

- `submission_repro_data/regenerated_paper_artifacts/`
