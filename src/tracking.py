"""Thin MLflow wrapper so every run carries the same lineage tags."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager

from src.config import PROJECT_ROOT, resolve_path, resolve_tracking_uri


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_info() -> dict[str, str]:
    """Commit plus whether tracked files differ from it (a dirty run is not reproducible).

    Without git (e.g. inside the pipeline Docker image, which has no ``.git``), fall back to the
    ``GIT_COMMIT`` / ``GIT_DIRTY`` environment variables baked in at image build time.
    """
    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if commit is None:
        return {
            "git_commit": os.environ.get("GIT_COMMIT") or "unknown",
            "git_dirty": os.environ.get("GIT_DIRTY") or "unknown",
        }
    return {
        "git_commit": commit,
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


@contextmanager
def start_run(cfg: dict, run_name: str, tags: dict | None = None, run_id: str | None = None):
    """Start a new MLflow run, or reopen ``run_id`` (resumed training, post-hoc analysis).

    New runs get lineage tags (git commit, dirty flag, Airflow run). A reopened run keeps the
    lineage it was created with, so later work never rewrites where a result came from; callers
    record their own provenance tags instead (e.g. ``resumed_at_commit``).
    """
    import mlflow

    tracking = cfg["tracking"]
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking["tracking_uri"]))
    name = tracking["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        location = resolve_path(tracking.get("artifact_location", "mlartifacts"))
        mlflow.create_experiment(name, artifact_location=location.as_uri())
    mlflow.set_experiment(name)

    if run_id:
        with mlflow.start_run(run_id=run_id) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run
        return

    lineage = git_info()
    if airflow_run := os.environ.get("AIRFLOW_CTX_DAG_RUN_ID"):
        lineage["airflow_dag_run_id"] = airflow_run
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**lineage, **(tags or {})})
        yield run
