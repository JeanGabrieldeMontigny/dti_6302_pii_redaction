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
    """Commit plus whether tracked files differ from it (a dirty run is not reproducible)."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_dirty": "unknown" if status is None else str(bool(status)).lower(),
    }


@contextmanager
def start_run(cfg: dict, run_name: str, tags: dict | None = None):
    import mlflow

    tracking = cfg["tracking"]
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking["tracking_uri"]))
    name = tracking["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        location = resolve_path(tracking.get("artifact_location", "mlartifacts"))
        mlflow.create_experiment(name, artifact_location=location.as_uri())
    mlflow.set_experiment(name)

    lineage = git_info()
    if airflow_run := os.environ.get("AIRFLOW_CTX_DAG_RUN_ID"):
        lineage["airflow_dag_run_id"] = airflow_run
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**lineage, **(tags or {})})
        yield run
