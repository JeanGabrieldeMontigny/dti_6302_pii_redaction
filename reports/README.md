# Reports

Project report, architecture diagram, and evidence.

- `validation/` – data-validation reports written by the pipeline (`make data` / `make validate`):
  `raw_validation.*` (Hugging Face data as downloaded) and `processed_validation.*` (clean splits).
  JSON for machines, Markdown for review. Regenerated on every run.
- `baseline/` – regex baseline evaluation on validation (`make baseline`): Markdown report,
  per-label CSV, and a JSON summary with the MLflow run id, `data_version` and git commit.
- `experiments/` – Keras tagger evaluations on validation (`make train` / `make experiments`), one
  set of files per experiment, same format as `baseline/`.
- `figures/` – figures for the report.

MLflow (`make mlflow`) holds the full run history; these files are the reviewable snapshot.
