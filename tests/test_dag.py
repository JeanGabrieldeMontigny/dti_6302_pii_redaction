"""DAG integrity: the Airflow DAG parses and wires the pipeline steps in order."""

import importlib.util
import sys

import pytest

from src.config import PROJECT_ROOT

if sys.platform == "win32":
    # Airflow does not run on native Windows (it needs fcntl), and a failed import leaves
    # SQLAlchemy mappers half-configured, which then breaks MLflow in later tests.
    # CI (Linux) runs it.
    pytest.skip("Airflow is not supported on native Windows", allow_module_level=True)
pytest.importorskip("airflow")

DAG_FILE = PROJECT_ROOT / "pipelines" / "dags" / "pii_redaction_pipeline.py"


def test_dag_structure():
    spec = importlib.util.spec_from_file_location("pii_dag", DAG_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dags = [v for v in vars(module).values() if type(v).__name__ == "DAG"]
    if not dags:  # @dag-decorated function registered via call; find it through the decorator
        dag = module.pii_redaction_pipeline()
    else:
        dag = dags[0]
    chain = ["ingest", "validate_raw", "preprocess", "validate_processed"]
    assert sorted(dag.task_ids) == sorted([*chain, "evaluate_baseline", "train"])
    for upstream, downstream in zip(chain, chain[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids
    # Baseline evaluation and training both depend only on validated processed data.
    assert dag.get_task("validate_processed").downstream_task_ids == {"evaluate_baseline", "train"}
    assert dag.get_task("ingest").retries == 3
    assert dag.get_task("validate_raw").retries == 0
    assert dag.get_task("train").retries == 1
