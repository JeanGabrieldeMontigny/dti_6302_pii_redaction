"""Airflow DAG for the PII redaction training pipeline.

Each task is a thin wrapper around a function in ``pipelines.steps``; data moves between tasks
through files on disk, and only small summaries go through XCom.

    ingest -> validate_raw -> preprocess -> validate_processed -+-> evaluate_baseline
                                                                +-> train

The training config is the DAG's ``config`` param, e.g. ``configs/experiments/e2_dilated_cnn.yaml``.

Failure behaviour:
- ``ingest`` touches the network, so it retries with exponential backoff.
- ``train`` retries once (transient resource failures); the retry resumes from the last
  completed epoch (per-epoch backups), in the same MLflow run.
- Validation steps never retry: a failed data check is deterministic and needs a human
  (fix upstream data or update the schema/thresholds), so the run stops there.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, get_current_context, task


def _cfg() -> dict:
    from src.config import load_config

    params = get_current_context()["params"]
    return load_config(params.get("config") or None)


@dag(
    dag_id="pii_redaction_pipeline",
    schedule=None,  # triggered manually / by CI; no fixed cadence for a static dataset
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["pii", "training"],
    default_args={"retries": 0, "execution_timeout": timedelta(hours=2)},
    params={
        "config": Param(
            "configs/default.yaml",
            type="string",
            description="Pipeline config, relative to the repository root",
        )
    },
)
def pii_redaction_pipeline():
    @task(retries=3, retry_delay=timedelta(seconds=30), retry_exponential_backoff=True)
    def ingest() -> dict:
        from pipelines import steps

        return steps.ingest(_cfg())

    @task
    def validate_raw(_upstream: dict) -> dict:
        from pipelines import steps

        return steps.validate_raw(_cfg())

    @task
    def preprocess(_upstream: dict) -> dict:
        from pipelines import steps

        return steps.preprocess(_cfg())

    @task
    def validate_processed(_upstream: dict) -> dict:
        from pipelines import steps

        return steps.validate_processed(_cfg())

    @task
    def evaluate_baseline(_upstream: dict) -> dict:
        from pipelines import steps

        return steps.evaluate_baseline(_cfg())

    # One retry for transient failures (out of memory, interrupted worker). The retry resumes from
    # the last completed epoch instead of starting over; a second failure needs a human.
    @task(retries=1, retry_delay=timedelta(minutes=1), execution_timeout=timedelta(hours=6))
    def train(_upstream: dict) -> dict:
        from pipelines import steps

        return steps.train(_cfg())

    validated = validate_processed(preprocess(validate_raw(ingest())))
    evaluate_baseline(validated)
    train(validated)


pii_redaction_pipeline()
