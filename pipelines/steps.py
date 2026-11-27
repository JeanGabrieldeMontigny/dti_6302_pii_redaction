"""Pipeline steps. Each step reads its inputs from disk, writes its outputs to disk and returns a
small JSON-serialisable summary, so the same functions run from the CLI or as Airflow tasks.

    ingest -> validate_raw -> preprocess -> validate_processed -> evaluate_baseline
                                                            \\-> train
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from src.config import resolve_path, resolve_tracking_uri
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


def source_sha256(path: Path) -> str:
    """SHA-256 of a source file with line endings normalised to LF.

    Git may check text files out with CRLF (Windows) or LF (macOS/Linux); the same commit must
    give the same ``data_version`` on every platform.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def data_version(cfg: dict) -> str:
    """Identifier for the processed data: raw file hashes, split settings and data-code hashes."""
    manifest = json.loads((_raw_dir(cfg) / ingest_mod.MANIFEST).read_text())
    code = {f: source_sha256(resolve_path(f)) for f in DATA_CODE}
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


SCALAR_METRICS = (
    "detection_recall",
    "char_precision",
    "char_recall",
    "type_macro_f1",
    "type_accuracy",
    "ms_per_doc",
)


def _evaluate_and_log(cfg: dict, model, df: pd.DataFrame, split: str, report_dir: Path) -> dict:
    """Evaluate ``model`` on ``df``, write the report files and log everything to the active run.

    Every model (rules or Keras) goes through this function, so all runs are directly comparable.
    """
    import mlflow

    from src.models.evaluate import evaluate_model, per_label_table, render_markdown

    ev = cfg["evaluation"]
    result = evaluate_model(model, df, ev["slice_columns"], ev["min_coverage"])
    report_md = render_markdown(result, model.name, split)
    per_label_csv = per_label_table(result).to_csv(index=False)

    # Reviewable evidence in the repo, alongside the MLflow run.
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{model.name}_{split}.md").write_text(report_md)
    (report_dir / f"{model.name}_{split}_per_label.csv").write_text(per_label_csv)

    mlflow.log_params({**model.params(), "min_coverage": ev["min_coverage"], "seed": cfg["seed"]})
    mlflow.log_params({f"data.{k}": v for k, v in cfg["data"].items() if k != "splits"})
    mlflow.log_metrics({f"{split}.{k}": result[k] for k in SCALAR_METRICS})
    mlflow.log_metrics(
        {f"{split}.recall.{lbl}": v["detection_recall"] for lbl, v in result["per_label"].items()}
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
    return result


def _write_summary(
    report_dir: Path,
    name: str,
    split: str,
    result: dict,
    run_id: str,
    manifest: dict,
    extra: dict | None = None,
) -> dict:
    """JSON summary next to the Markdown report, with the MLflow run id and lineage tags.

    ``reports/experiments/compare.md`` is built from these files (``compare`` step).
    """
    import mlflow

    tags = mlflow.get_run(run_id).data.tags
    summary = {k: v for k, v in result.items() if k not in ("per_label", "slices")}
    summary |= {
        "run_id": run_id,
        "tracking_uri": mlflow.get_tracking_uri(),
        "data_version": manifest["data_version"],
        **(extra or {}),
        **tags,
    }
    summary = {k: v for k, v in summary.items() if not k.startswith("mlflow.")}
    (report_dir / f"{name}_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


def _headline(result: dict) -> dict:
    return {k: round(result[k], 4) for k in ("detection_recall", "char_precision", "type_macro_f1")}


def evaluate_baseline(cfg: dict, split: str = "validation") -> dict:
    """Evaluate the regex baseline and log it to MLflow (no training needed)."""
    import mlflow

    from src.models.regex_baseline import RegexBaseline
    from src.tracking import start_run

    manifest = check_processed_current(cfg)
    model = RegexBaseline()
    df = load_processed(cfg, [split])[split]
    report_dir = resolve_path(cfg["evaluation"]["report_dir"])

    with start_run(
        cfg,
        run_name=model.name,
        tags={
            "data_version": manifest["data_version"],
            "model_family": "rules",
            "eval_split": split,
        },
    ) as run:
        result = _evaluate_and_log(cfg, model, df, split, report_dir)
        mlflow.log_dict(manifest, "data/manifest.json")
        run_id = run.info.run_id

    _write_summary(report_dir, model.name, split, result, run_id, manifest)
    return {"run_id": run_id, "model": model.name, "split": split, **_headline(result)}


def train(cfg: dict, split: str = "validation") -> dict:
    """Train the configured Keras tagger, evaluate it on validation and save a versioned model.

    Output: ``artifacts/models/<model_version>/`` with ``model.keras``, ``preprocessing.json``
    (everything inference needs: vocabulary, labels, windowing, decoding) and ``model_card.json``.
    The same directory is logged to MLflow under ``model/``.

    Pause / resume: the training state is backed up after every epoch in
    ``artifacts/checkpoints/<checkpoint_key>/``. If the process stops (Ctrl+C, crash, reboot),
    running the same command again continues from the last completed epoch, in the same MLflow
    run and with the same ``model_version``. Set ``train.resume: false`` to start over instead.
    The backup is deleted once the run has completed.
    """
    import mlflow

    from src.data.schema import LABELS
    from src.features.vocab import CharVocab
    from src.models.predict import KerasTagger, preprocessing_spec
    from src.models.train import (
        checkpoint_key,
        hardware_info,
        mlflow_epoch_logger,
        progress_reporter,
        subset_by_uid,
        train_model,
    )
    from src.tracking import git_info, start_run

    manifest = check_processed_current(cfg)
    data = load_processed(cfg, ["train", split])
    feat, mcfg, tcfg = cfg["features"], cfg["model"], cfg["train"]
    train_df = subset_by_uid(data["train"], tcfg["train_fraction"], cfg["seed"])
    # Fit on the training data actually used, so a subset experiment never sees other text.
    vocab = CharVocab.fit(train_df["text"], feat["min_char_freq"])
    spec = preprocessing_spec(vocab, LABELS, feat["window"], feat["stride"], cfg["decode"])
    report_dir = resolve_path(tcfg["report_dir"])

    ckpt_dir = (
        resolve_path(cfg["artifacts"]["dir"])
        / "checkpoints"
        / checkpoint_key(cfg, manifest["data_version"])
    )
    resume = _resumable_run(ckpt_dir, enabled=tcfg.get("resume", True))
    git = git_info()

    with start_run(
        cfg,
        run_name=mcfg["name"],
        tags={
            "data_version": manifest["data_version"],
            "model_family": "keras",
            "eval_split": split,
        },
        run_id=resume["run_id"] if resume else None,
    ) as run:
        run_id = run.info.run_id
        model_version = f"{mcfg['name']}-{run_id[:8]}"
        if resume:
            mlflow.set_tags({"resumed": "true", "resumed_at_commit": git["git_commit"]})
            if resume["git_commit"] != git["git_commit"]:
                log.warning(
                    "Resuming a run started at commit %s from commit %s",
                    resume["git_commit"],
                    git["git_commit"],
                )
                mlflow.set_tag("resumed_across_commits", "true")
        else:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tag("hardware", hardware_info())
            (ckpt_dir / "run.json").write_text(
                json.dumps({"run_id": run_id, "git_commit": git["git_commit"]})
            )
            mlflow.set_tag("model_version", model_version)
            mlflow.log_params({f"model.{k}": v for k, v in mcfg.items()})
            mlflow.log_params({f"train.{k}": v for k, v in tcfg.items() if k != "report_dir"})
            mlflow.log_params({f"features.{k}": v for k, v in feat.items()})
            mlflow.log_params(
                {"train_docs": len(train_df), "train_uids": train_df["uid"].nunique()}
            )

        t0 = time.perf_counter()
        model, history, info = train_model(
            cfg,
            train_df,
            data[split],
            vocab,
            LABELS,
            callbacks=[
                mlflow_epoch_logger(),
                progress_reporter(ckpt_dir / "progress.json", model_version),
            ],
            backup_dir=ckpt_dir / "backup",
        )
        train_seconds = time.perf_counter() - t0
        # Count epochs from MLflow: Keras' history only covers this process (not resumed epochs).
        epochs_run = len(mlflow.MlflowClient().get_metric_history(run_id, "epoch.loss"))
        mlflow.log_params({**info, "epochs_run": epochs_run})

        tagger = KerasTagger(model, spec, name=mcfg["name"])
        result = _evaluate_and_log(cfg, tagger, data[split], split, report_dir)
        sweep_best = _decode_sweep_and_log(cfg, tagger, data[split], split, report_dir)

        model_dir = resolve_path(cfg["artifacts"]["dir"]) / "models" / model_version
        tagger.save(model_dir)
        size_bytes = (model_dir / "model.keras").stat().st_size
        mlflow.log_metrics(
            {
                "n_params": model.count_params(),
                "model_size_bytes": size_bytes,
                "train_seconds": train_seconds,
            }
        )
        card = {
            "model_version": model_version,
            "mlflow_run_id": run_id,
            "data_version": manifest["data_version"],
            **git,
            "architecture": mcfg,
            "n_params": model.count_params(),
            "model_size_bytes": size_bytes,
            "vocab_sha256": spec["vocab_sha256"],
            "train": {**tcfg, **info, "epochs_run": epochs_run, "resumed": bool(resume)},
            "eval_split": split,
            "metrics": {k: result[k] for k in SCALAR_METRICS},
            "decode_sweep_best": sweep_best,
        }
        (model_dir / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
        mlflow.log_artifacts(str(model_dir), "model")
        mlflow.log_dict(manifest, "data/manifest.json")

    # Completed: nothing left to resume. (Keras already removed the weights backup.)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    _write_summary(
        report_dir,
        mcfg["name"],
        split,
        result,
        run_id,
        manifest,
        extra={
            "model_version": model_version,
            "n_params": model.count_params(),
            "model_size_bytes": size_bytes,
            "train_seconds": train_seconds,
            "epochs_run": epochs_run,
            "decode_sweep_best": sweep_best,
        },
    )
    return {
        "run_id": run_id,
        "model_version": model_version,
        "model_dir": str(model_dir),
        "split": split,
        **_headline(result),
    }


def _decode_sweep_and_log(
    cfg: dict, tagger, df: pd.DataFrame, split: str, report_dir: Path, log_to_mlflow: bool = True
) -> dict | None:
    """Score every decoding setting in ``evaluation.decode_sweep`` and log the curve to the active
    MLflow run. Returns the recommended setting (best recall meeting the precision target)."""
    import mlflow

    from src.models.sweep import decode_sweep, plot_sweep, recommend

    sw = cfg["evaluation"]["decode_sweep"]
    table = decode_sweep(
        tagger, df, sw["thresholds"], sw["word_snap"], cfg["evaluation"]["min_coverage"]
    )
    best = recommend(table, sw["min_char_precision"])
    csv_path = report_dir / f"{tagger.name}_decode_sweep.csv"
    png_path = report_dir / f"{tagger.name}_decode_sweep.png"
    report_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(csv_path, index=False)
    plot_sweep(
        table,
        png_path,
        title=f"{tagger.name}: recall vs precision by decoding setting ({split})",
        current=(tagger.threshold, tagger.word_snap),
        recommended=best,
        min_char_precision=sw["min_char_precision"],
    )
    log.info("Decode sweep: %s", best or "no setting meets the precision target")
    if not log_to_mlflow:
        return best
    mlflow.log_artifact(str(csv_path), "eval")
    mlflow.log_artifact(str(png_path), "eval")
    if best:
        mlflow.log_metrics(
            {
                f"{split}.sweep_best.detection_recall": best["detection_recall"],
                f"{split}.sweep_best.char_precision": best["char_precision"],
                f"{split}.sweep_best.threshold": best["threshold"],
                f"{split}.sweep_best.word_snap": float(best["word_snap"]),
            }
        )
    return best


def decode_sweep(cfg: dict, split: str = "validation") -> dict:
    """Run the decoding sweep for an already trained model of the configured experiment, and
    attach the results to its MLflow run and report summary.

    The model is the one behind the experiment's current report (``<name>_<split>.json``), so the
    sweep always matches the committed evidence; without a report, the newest model is used.
    """
    import mlflow

    from src.models.predict import KerasTagger, find_model_dir
    from src.tracking import git_info, start_run

    check_processed_current(cfg)
    name = cfg["model"]["name"]
    models_root = resolve_path(cfg["artifacts"]["dir"]) / "models"
    report_dir = resolve_path(cfg["train"]["report_dir"])
    summary_path = report_dir / f"{name}_{split}.json"
    reported = None
    if summary_path.exists():
        run = json.loads(summary_path.read_text())["run_id"]
        reported = models_root / f"{name}-{run[:8]}"  # model_version = <name>-<run id prefix>
    if reported is not None and reported.exists():
        model_dir = reported
    else:
        if reported is not None:
            log.warning("Model %s from the report is not on this machine", reported.name)
        model_dir = find_model_dir(models_root, name + "-")
    log.info("Decode sweep for model %s", model_dir.name)
    card = json.loads((model_dir / "model_card.json").read_text())
    tagger = KerasTagger.load(model_dir, name=name)
    df = load_processed(cfg, [split])[split]

    run_id = card["mlflow_run_id"]
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["tracking"]["tracking_uri"]))
    try:
        mlflow.get_run(run_id)
        in_store = True
    except mlflow.exceptions.MlflowException:  # trained against another store (e.g. Docker)
        log.warning("MLflow run %s is not in this tracking store; writing reports only", run_id)
        in_store = False

    if in_store:
        git = git_info()
        sweep_tags = {
            "decode_sweep_commit": git["git_commit"],
            "decode_sweep_dirty": git["git_dirty"],
        }
        with start_run(cfg, run_name="", run_id=run_id, tags=sweep_tags):
            best = _decode_sweep_and_log(cfg, tagger, df, split, report_dir)
    else:
        best = _decode_sweep_and_log(cfg, tagger, df, split, report_dir, log_to_mlflow=False)

    card["decode_sweep_best"] = best
    (model_dir / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("run_id") == run_id:
            summary["decode_sweep_best"] = best
            # Summaries written before these fields existed get them from the model card / run.
            backfill = {
                "model_version": card["model_version"],
                "n_params": card.get("n_params"),
                "model_size_bytes": card.get("model_size_bytes"),
                "epochs_run": card.get("train", {}).get("epochs_run"),
            }
            if in_store:
                run = mlflow.get_run(run_id)
                backfill["train_seconds"] = run.data.metrics.get("train_seconds")
                backfill["hardware"] = run.data.tags.get("hardware")
            for key, value in backfill.items():
                if value is not None:
                    summary.setdefault(key, value)
            summary_path.write_text(json.dumps(summary, indent=2))
    return {"model_version": model_dir.name, "split": split, "best": best}


def compare(cfg: dict, split: str = "validation") -> dict:
    """Build reports/experiments/comparison.{md,csv} from every run's committed summary files."""
    from src.models.compare import build_comparison

    out = build_comparison(
        baseline_dir=resolve_path(cfg["evaluation"]["report_dir"]),
        experiments_dir=resolve_path(cfg["train"]["report_dir"]),
        split=split,
        min_char_precision=cfg["evaluation"]["decode_sweep"]["min_char_precision"],
    )
    log.info("Comparison written to %s", out["markdown"])
    return out


def _resumable_run(ckpt_dir: Path, enabled: bool) -> dict | None:
    """The interrupted run to continue (``{"run_id", "git_commit"}``), or None to start fresh."""
    meta = ckpt_dir / "run.json"
    if enabled and meta.exists():
        info = json.loads(meta.read_text())
        log.info("Resuming interrupted MLflow run %s from %s", info["run_id"], ckpt_dir)
        return info
    if ckpt_dir.exists():
        log.info("Discarding previous training backup %s (train.resume is off)", ckpt_dir)
        shutil.rmtree(ckpt_dir)
    return None


def _metric_key(value: str) -> str:
    """MLflow metric names allow only alphanumerics, _ - . / and spaces."""
    return "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in value)


STEPS = {
    "ingest": ingest,
    "validate_raw": validate_raw,
    "preprocess": preprocess,
    "validate_processed": validate_processed,
    "evaluate_baseline": evaluate_baseline,
    "train": train,
}
# Steps run on request only (not part of `all`).
EXTRA_STEPS = {
    "decode_sweep": decode_sweep,
    "compare": compare,
}
