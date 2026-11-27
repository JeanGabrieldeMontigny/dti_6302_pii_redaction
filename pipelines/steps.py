"""Pipeline steps. Each step reads its inputs from disk, writes its outputs to disk and returns a
small JSON-serialisable summary, so the same functions run from the CLI or as Airflow tasks.

    ingest -> validate_raw -> preprocess -> validate_processed -> evaluate_baseline
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import resolve_path
from src.data import ingest as ingest_mod
from src.data.io import read_split, write_split
from src.data.preprocess import preprocess as preprocess_df
from src.data.split import leakage_report, remove_cross_split_leakage, split_train_val
from src.data.validate import validate_processed as validate_processed_df
from src.data.validate import validate_raw as validate_raw_df
from src.data.validate import write_report

log = logging.getLogger(__name__)

PROCESSED_SPLITS = ("train", "validation", "test")


def _raw_dir(cfg) -> Path:
    return resolve_path(cfg["data"]["raw_dir"])


def _processed_dir(cfg) -> Path:
    return resolve_path(cfg["data"]["processed_dir"])


def _report_dir(cfg) -> Path:
    return resolve_path(cfg["validation"]["report_dir"])


# Source files whose contents determine the processed data. Their hashes feed ``data_version``,
# so any change to preprocessing or splitting logic produces a new version automatically.
DATA_CODE = ("src/data/schema.py", "src/data/preprocess.py", "src/data/split.py")


def data_version(cfg: dict) -> str:
    """Identifier for the processed data: raw file hashes, split settings and data-code hashes."""
    manifest = json.loads((_raw_dir(cfg) / ingest_mod.MANIFEST).read_text())
    code = {f: ingest_mod.sha256_file(resolve_path(f)) for f in DATA_CODE}
    key = json.dumps(
        {
            "files": manifest["files"],
            "val_fraction": cfg["data"]["val_fraction"],
            "seed": cfg["seed"],
            "code": code,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def check_processed_current(cfg: dict) -> dict:
    """Return the processed manifest, failing if it was built from other data, settings or code."""
    path = _processed_dir(cfg) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the preprocess step first")
    manifest = json.loads(path.read_text())
    ingest_mod.verify_manifest(_raw_dir(cfg), cfg["data"]["splits"])
    expected = data_version(cfg)
    if manifest["data_version"] != expected:
        raise RuntimeError(
            f"processed data is version {manifest['data_version']} but the current raw data, "
            f"config and code give {expected}; re-run preprocess (make data)"
        )
    for name, digest in manifest.get("files", {}).items():
        if ingest_mod.sha256_file(_processed_dir(cfg) / name) != digest:
            raise RuntimeError(f"processed file {name} changed since preprocessing; re-run it")
    return manifest


# --------------------------------------------------------------------------- steps


def ingest(cfg: dict) -> dict:
    d = cfg["data"]
    manifest = ingest_mod.download_raw(d["hf_repo"], d["hf_revision"], d["splits"], _raw_dir(cfg))
    return {"raw_dir": str(_raw_dir(cfg)), "revision": manifest["revision"]}


def _read_raw(cfg: dict) -> dict[str, pd.DataFrame]:
    """Read raw splits after checking they are the files recorded by ``ingest``."""
    ingest_mod.verify_manifest(_raw_dir(cfg), cfg["data"]["splits"])
    return {
        s: pd.read_parquet(_raw_dir(cfg) / ingest_mod.raw_filename(s))
        for s in cfg["data"]["splits"]
    }


def validate_raw(cfg: dict) -> dict:
    report = validate_raw_df(_read_raw(cfg), cfg["validation"])
    json_path, md_path = write_report(report, _report_dir(cfg))
    log.info("Raw validation report: %s", md_path)
    report.raise_if_failed()
    return {
        "report": str(json_path),
        "warnings": [f"{c.split}/{c.name}" for c in report.failed_warnings],
    }


def preprocess(cfg: dict) -> dict:
    """Clean each raw split, carve validation out of train by uid, and remove cross-split leaks."""
    raw = _read_raw(cfg)
    stats, clean = {}, {}
    for name, df in raw.items():
        clean[name], s = preprocess_df(df)
        stats[name] = s.as_dict()

    train, validation = split_train_val(clean["train"], cfg["data"]["val_fraction"], cfg["seed"])
    splits, removed = remove_cross_split_leakage(
        {"train": train, "validation": validation, "test": clean["test"]}, reference="test"
    )

    out_dir = _processed_dir(cfg)
    for name in PROCESSED_SPLITS:
        write_split(splits[name], out_dir / f"{name}.parquet")

    manifest = {
        "data_version": data_version(cfg),
        "counts": {name: len(splits[name]) for name in PROCESSED_SPLITS},
        "files": {
            f"{name}.parquet": ingest_mod.sha256_file(out_dir / f"{name}.parquet")
            for name in PROCESSED_SPLITS
        },
        "preprocess_stats": stats,
        "removed_for_leakage": removed,
        "leakage": leakage_report(splits),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Processed splits written to %s: %s", out_dir, manifest["counts"])
    return {"processed_dir": str(out_dir), **{k: manifest[k] for k in ("data_version", "counts")}}


def load_processed(cfg: dict, names=PROCESSED_SPLITS) -> dict[str, pd.DataFrame]:
    return {n: read_split(_processed_dir(cfg) / f"{n}.parquet") for n in names}


def validate_processed(cfg: dict) -> dict:
    check_processed_current(cfg)
    report = validate_processed_df(
        load_processed(cfg), cfg["validation"], val_fraction=cfg["data"]["val_fraction"]
    )
    json_path, md_path = write_report(report, _report_dir(cfg))
    log.info("Processed validation report: %s", md_path)
    report.raise_if_failed()
    return {
        "report": str(json_path),
        "warnings": [f"{c.split}/{c.name}" for c in report.failed_warnings],
    }


def evaluate_baseline(cfg: dict, split: str = "validation") -> dict:
    """Evaluate the regex baseline and log it to MLflow (no training needed)."""
    import mlflow

    from src.models.evaluate import evaluate_model, per_label_table, render_markdown
    from src.models.regex_baseline import RegexBaseline

    manifest = check_processed_current(cfg)
    model = RegexBaseline()
    df = load_processed(cfg, [split])[split]
    ev = cfg["evaluation"]
    result = evaluate_model(model, df, ev["slice_columns"], ev["min_coverage"])
    report_md = render_markdown(result, model.name, split)
    per_label_csv = per_label_table(result).to_csv(index=False)

    # Reviewable evidence in the repo, alongside the MLflow run.
    report_dir = resolve_path(ev["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{model.name}_{split}.md").write_text(report_md)
    (report_dir / f"{model.name}_{split}_per_label.csv").write_text(per_label_csv)
    summary = {k: v for k, v in result.items() if k not in ("per_label", "slices")}

    from src.tracking import start_run

    with start_run(
        cfg,
        run_name=model.name,
        tags={
            "data_version": manifest["data_version"],
            "model_family": "rules",
            "eval_split": split,
        },
    ) as run:
        mlflow.log_params(
            {**model.params(), "min_coverage": ev["min_coverage"], "seed": cfg["seed"]}
        )
        mlflow.log_params({f"data.{k}": v for k, v in cfg["data"].items() if k != "splits"})
        scalar = [
            "detection_recall",
            "char_precision",
            "char_recall",
            "type_macro_f1",
            "type_accuracy",
            "ms_per_doc",
        ]
        mlflow.log_metrics({f"{split}.{k}": result[k] for k in scalar})
        mlflow.log_metrics(
            {
                f"{split}.recall.{lbl}": v["detection_recall"]
                for lbl, v in result["per_label"].items()
            }
        )
        for col, groups in result["slices"].items():
            mlflow.log_metrics(
                {
                    f"{split}.recall.{col}.{_metric_key(k)}": m["detection_recall"]
                    for k, m in groups.items()
                }
            )
        mlflow.log_dict(result, f"eval/{split}_metrics.json")
        mlflow.log_text(report_md, f"eval/{split}_report.md")
        mlflow.log_text(per_label_csv, f"eval/{split}_per_label.csv")
        mlflow.log_dict(manifest, "data/manifest.json")
        run_id = run.info.run_id

    tags = mlflow.get_run(run_id).data.tags
    summary |= {"run_id": run_id, "data_version": manifest["data_version"], **tags}
    summary = {k: v for k, v in summary.items() if not k.startswith("mlflow.")}
    (report_dir / f"{model.name}_{split}.json").write_text(json.dumps(summary, indent=2))

    return {
        "run_id": run_id,
        "model": model.name,
        "split": split,
        **{k: round(result[k], 4) for k in ("detection_recall", "char_precision", "type_macro_f1")},
    }


def _metric_key(value: str) -> str:
    """MLflow metric names allow only alphanumerics, _ - . / and spaces."""
    return "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in value)


STEPS = {
    "ingest": ingest,
    "validate_raw": validate_raw,
    "preprocess": preprocess,
    "validate_processed": validate_processed,
    "evaluate_baseline": evaluate_baseline,
}
