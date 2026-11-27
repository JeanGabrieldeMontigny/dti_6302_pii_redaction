# How to run the project

Step-by-step guide to what currently works on the `cp3-keras-baseline` branch: the data pipeline,
regex baseline, Keras model training, MLflow, Airflow (in Docker, on any OS), the pipeline Docker
image, and tests.

Not built yet: the browser demo app, the monitoring service and model export. `make serve`,
`make evaluate` and `make monitor` are still placeholders.

---

## 0. Quick start

Every task has the same name on both platforms:

| | macOS / Linux | Windows |
| --- | --- | --- |
| Run a task | `make <task>` | `scripts\run.cmd <task>` |
| With a config | `make train CONFIG=configs/experiments/e1_small_cnn_subset.yaml` | `scripts\run.cmd train -Config configs/experiments/e1_small_cnn_subset.yaml` |
| List tasks | `make help` | `scripts\run.cmd help` |

`scripts\run.cmd` works from Command Prompt and PowerShell (in PowerShell you can also type
`.\scripts\run.cmd`). It needs no execution-policy changes.

```text
setup → hooks → data → baseline → train (E1) → compare → mlflow    the core workflow, ~5 min
status                                       progress + ETA of a running training
redact -Text "..."                           try a trained model on your own text
train-gpu                                    the same training on an NVIDIA GPU (Docker)
airflow-docker-up → airflow-docker-run       the pipeline orchestrated by Airflow
test, lint                                   quality checks (same as CI)
```

---

## 1. Prerequisites

| Tool | Needed for | Install |
| --- | --- | --- |
| Git | cloning | <https://git-scm.com> |
| [uv](https://docs.astral.sh/uv/) | Python + dependencies (it downloads Python 3.12 itself) | macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh` · Windows: `winget install astral-sh.uv` |
| Docker Desktop | Airflow (required on Windows), pipeline image, GPU training | <https://www.docker.com/products/docker-desktop/>. **Start it before any `docker`/`airflow-docker`/`train-gpu` task** |
| NVIDIA GPU + driver | optional: `train-gpu` (section 12) | a recent NVIDIA driver on the host; nothing else (CUDA comes in the image) |
| `make` | macOS/Linux shortcuts | preinstalled. Windows uses `scripts\run.cmd` instead |

**Disk:** ~2.5 GB for the Python environment, ~0.4 GB for the data, ~16 GB for the pipeline and
Airflow Docker images (TensorFlow is large), plus ~12 GB for the GPU image if you use it.
**Hardware:** CPU only is fine. A GPU only makes training faster (section 12).

> **Airflow on Windows:** Airflow cannot run directly on Windows, even when `pip`/`uv` installed
> it (it needs the Unix-only `fcntl` module). Run it in Docker (section 10), which works the same
> on every OS.

---

## 2. Get the code and install

```bash
git clone https://github.com/JeanGabrieldeMontigny/dti_6302_pii_redaction.git
cd dti_6302_pii_redaction
git checkout cp3-keras-baseline

make setup                     # Windows: scripts\run.cmd setup
```

`setup` runs `uv sync`. It creates `.venv/` and installs the exact locked versions from
`uv.lock`. The first time takes a few minutes. All tasks run inside this environment.

Then, once per clone, install the git hooks:

```bash
make hooks                     # Windows: scripts
un.cmd hooks
```

From then on every `git commit` first runs ruff (format + lint, auto-fixing what it can) and
blocks committing large files (> 1 MB: data, models, MLflow runs), merge-conflict markers, broken
YAML/TOML and private keys. If a hook fixed something, the commit stops: review the change,
`git add` it and commit again. Run all hooks by hand with `uv run pre-commit run --all-files`.

> The hooks need **Git 2.31 or newer** (`git --version`). With an older Git, every commit fails
> with `unknown option 'deduplicate'`. Upgrade Git: Windows `winget upgrade Git.Git`, macOS
> `brew upgrade git`.

---

## 3. Task reference

| Task | What it does | Time* |
| --- | --- | --- |
| `setup` | install dependencies | few min (first time) |
| `data` | download → validate → split → validate (section 4) | ~1 min + download |
| `validate` | re-run both validation stages | ~30 s |
| `baseline` | regex baseline on validation, logged to MLflow (section 5) | ~10 s |
| `train` | train + evaluate one Keras model, with `CONFIG` / `-Config` (section 6) | E1: ~40 s · E2: see section 6 |
| `experiments` | `train` for every file in `configs/experiments/` | sum of the above |
| `status` | epoch, step, losses and ETA of running (or interrupted) trainings (section 6) | instant |
| `decode-sweep` | recall/precision curve for the latest model of `CONFIG` (section 6) | ~1–3 min |
| `compare` | `reports/experiments/comparison.md`: every model side by side (section 7) | instant |
| `redact` | redact `TEXT` / `-Text` with a trained model (section 8) | ~5 s |
| `train-gpu` | `train` on an NVIDIA GPU in Docker (section 12) | E2: ~7.6 min/epoch (RTX 5090) vs ~11 on CPU |
| `hooks` | install the git pre-commit hooks (section 2) | |
| `pipeline` | every step without Airflow. Trains with `CONFIG` (**default config = slow full model**) | |
| `mlflow` | MLflow UI for local runs, <http://localhost:5000> (section 7) | |
| `test` / `lint` / `format` | tests, lint + format check (same as CI), auto-format (section 9) | ~15 s |
| `airflow-docker-up` | start Airflow + its MLflow UI in Docker (section 10) | ~5 min first build, then seconds |
| `airflow-docker-run` | trigger one DAG run with `CONFIG` and follow it until it finishes | E1: ~2–3 min |
| `airflow-docker-logs` / `airflow-docker-down` | follow logs / stop Airflow | |
| `docker-build` / `docker-train` | pipeline image: build it, train `CONFIG` inside it (section 11) | ~4–6 min (first build) / ~1–2 min (E1) |
| `airflow` / `airflow-run` | native Airflow, **macOS/Linux only** | |

\*Measured on a 16-core desktop CPU. Expect slower on a laptop.

**Order matters:** run `data` before `baseline`/`train`. They refuse to run on missing or stale
data.

---

## 4. Prepare the data: `data`

Runs four pipeline steps:

1. **ingest**: downloads NVIDIA Nemotron-PII from Hugging Face at the commit pinned in
   `configs/default.yaml` into `data/raw/` (~300 MB) and records a SHA-256 per file. Skipped when
   the files are already there and match.
2. **validate_raw**: schema, missing-value, domain and statistical checks. Writes
   `reports/validation/raw_validation.{md,json}` and **stops the pipeline** on failure.
3. **preprocess**: cleans spans and builds leakage-free train / validation / test splits (10% of
   personas → validation, assigned by hash). Writes `data/processed/` + `manifest.json`.
4. **validate_processed**: re-checks the clean splits, including zero overlap between splits.

**Expected:** 90,070 / 9,915 / 99,990 docs, `data_version` **`658e29cded93`**, one expected
warning (`drift.label_distribution_train_vs_test`), and `git status` shows **no changes** to
`reports/validation/`: same pinned data + same code gives the same output.

---

## 5. Regex baseline: `baseline`

Evaluates the hand-written regex rules on validation and logs an MLflow run. Expected detection
recall: **0.329** (char precision 0.993).

> This rewrites the committed `reports/baseline/` files (new run id, timing, and the order of
> tied rows). The metrics are identical. Discard with `git checkout reports/baseline` unless you
> mean to update the evidence.

---

## 6. Train a Keras model: `train`

```bash
make train CONFIG=configs/experiments/e1_small_cnn_subset.yaml
scripts\run.cmd train -Config configs/experiments/e1_small_cnn_subset.yaml      # Windows
```

| Experiment | Config | What it is | Validation result |
| --- | --- | --- | --- |
| E1 | `e1_small_cnn_subset.yaml` | 3-layer char CNN, 10% of training data, 3 epochs ("small model" baseline from the proposal) | recall **0.5446**, precision 0.9447, macro-F1 0.5839 (~40 s) |
| E2 | `e2_dilated_cnn.yaml` | residual dilated CNN, full training data, up to 10 epochs (early stopping) | recall **0.9486**, precision 0.9845, macro-F1 0.9805 (2 h 02 min on CPU, 10 epochs; best epoch 8 kept) |

What happens: check the data is current → fit the character vocabulary on the training data used
→ train (one log line per epoch with `loss` / `val_loss`, plus a progress line every 2 minutes)
→ evaluate on validation **with the same code and metrics as the regex baseline** → **decoding
sweep** → save a versioned model → log everything to MLflow.

| Output | Content |
| --- | --- |
| `artifacts/models/<model_version>/model.keras` | the trained model |
| `artifacts/models/<model_version>/preprocessing.json` | vocabulary, label order, window size, decoding settings: everything needed to use the model |
| `artifacts/models/<model_version>/model_card.json` | MLflow run id, `data_version`, git commit, metrics, size |
| `reports/experiments/<experiment>_validation.{md,json}` + `_per_label.csv` | evaluation report (committed as evidence) |
| `reports/experiments/<experiment>_decode_sweep.{png,csv}` | recall vs precision for each decoding setting (see below) |

`<model_version>` = experiment name + start of the MLflow run id, e.g. `e1_small_cnn_subset-faeb83d7`.

**New experiments:** copy a file in `configs/experiments/`, change values, and set `model.name` to
the new file's name (a test enforces this). Each file `extends: ../default.yaml` and lists only
what it changes.

> **Evidence runs:** commit first. Runs made with uncommitted changes to tracked files are tagged
> `git_dirty=true` and shouldn't be used in the report. Like `baseline`, `train` rewrites
> `reports/experiments/<experiment>_*`. Discard with `git checkout reports/experiments` when it
> was only a trial.

### Watching progress: `status`

Keras prints one line per epoch, which can be 10+ minutes apart. From any other terminal:

```bash
make status                    # Windows: scripts
un.cmd status
```

```text
e2_dilated_cnn-1a2b3c4d
  epoch 9/10, step 1234/3923  (83% overall), current loss 0.0247
  last completed epoch: loss 0.0252, val_loss 0.0279
  epoch time: ~11m37s
  ETA: 22m10s (around 15:38)  [updated 12s ago]
```

The training also logs the same information every 2 minutes. If a run stopped (crash, reboot,
closed terminal), `status` says **NOT RUNNING** and how to resume it.

### Decoding sweep: the recall/precision trade-off

How the model's per-character probabilities become redactions is a setting, not something
learned. **Threshold:** redact a character when P(PII) ≥ t. **Word snap:** extend a partial
redaction to the whole word. After training, every combination in
`evaluation.decode_sweep` is scored on validation, **without retraining**:

- `reports/experiments/<experiment>_decode_sweep.png`: detection recall vs char precision, one
  line per word-snap setting. The circle is the configured setting, and the star is the best
  recall that still meets the precision target (`min_char_precision`, 0.85 as proposed in the
  plan).
- The best setting is also in the model card, the summary JSON, MLflow (`validation.sweep_best.*`)
  and `comparison.md`.

For a model trained before the sweep existed, or to re-run it after changing the settings:

```bash
make decode-sweep CONFIG=configs/experiments/e1_small_cnn_subset.yaml
scripts
un.cmd decode-sweep -Config configs/experiments/e1_small_cnn_subset.yaml
```

> Choosing a threshold on validation and reporting validation results at that threshold is
> slightly optimistic. The held-out **test** split (used once, for the final model) gives the
> unbiased number.

### Pausing and resuming training

Long runs (E2 takes ~2 h on CPU) can be stopped and continued later:

1. **Stop** it any time: `Ctrl+C`, close the terminal, or even a crash or reboot.
2. **Resume** by running **exactly the same command** again. You'll see
   `Resuming interrupted MLflow run <id>`, and training continues from the **last completed
   epoch**. The epoch in progress when you stopped is redone.

How it works: after every epoch, the full training state (weights, optimizer, epoch number) is
saved in `artifacts/checkpoints/<experiment>-<hash>/`. The hash covers the experiment's
settings, seed and `data_version`, so a backup is only ever resumed by the same training.
Changing a setting starts a fresh run instead.

- The resumed run **continues the same MLflow run** (tag `resumed=true`) and keeps the same
  `model_version`. Its epoch curves show every epoch once.
- The backup folder is deleted automatically when the run completes.
- **Start over instead of resuming:** delete `artifacts/checkpoints/<experiment>-<hash>/`, or set
  `train.resume: false` in the config.
- In Airflow, the `train` task's automatic retry also resumes instead of starting from scratch.

Limits, and what to say in the report:

- A resumed run is **close to, but not bit-identical with**, an uninterrupted one. The data
  shuffle order restarts after a resume. Tested with E1 killed after epoch 1: recall 0.5472 vs
  0.5446 uninterrupted.
- Early stopping's "best epoch" tracking starts again at the resume point.
- If the code changed between the stop and the resume, the run gets the tag
  `resumed_across_commits=true`. Don't use such runs as evidence.

---

## 7. Comparing models: `compare` and MLflow

### The comparison report

```bash
make compare                   # Windows: scripts
un.cmd compare
```

This writes `reports/experiments/comparison.md` (and `.csv`), built from the committed report files.
It works the same whether a model was trained locally, in Airflow or on the GPU.

- **Overall table:** one row per model (regex baseline + each experiment's latest run) with
  recall, precision, macro-F1, speed, parameters, size, training time, epochs, hardware, the best
  decoding setting, and an **evidence** column (`yes`, or why the run can't be cited:
  uncommitted code, other data, ...).
- **Per-label table:** detection recall of every model for each of the 55 labels, best in bold.

Use it for the report's experiment table and slide 5. Re-run it after each training.

### MLflow UI: `mlflow`

Open <http://localhost:5000> → experiment **pii-redaction**. Stop with `Ctrl+C`. The data lives
in `mlflow.db` + `mlartifacts/` (git-ignored, local to your machine).

### What to look for

1. **Runs table:** click **Columns** and add `validation.detection_recall`,
   `validation.char_precision`, `validation.type_macro_f1`, `validation.ms_per_doc`, and the tags
   `git_dirty` and `model_version`. This is the experiment comparison table for the report and
   slides.
2. **Compare** (select `regex_baseline` + one `e1_small_cnn_subset` run → **Compare**): the
   recall/precision trade-off. Regex: precise but misses most PII (0.329 / 0.993). E1: catches far
   more, slightly less precise (0.545 / 0.945). Use the scatter plot with recall vs precision.
3. **Per-label recall** (`validation.recall.<label>`). Find three groups:
   - where the model wins (`bank_routing_number` 0 → 0.996, `vehicle_identifier` 0 → 0.98);
   - the few labels where regex is still slightly better;
   - labels both still miss (`occupation`, `gender`, `education_level`, ...). These are the
     targets for the next experiments.
4. **Slices** (`validation.recall.locale.*`, `.document_format.*`, `.domain.*`): check whether a
   group does much worse than the average. That is the rubric's slice analysis.
5. **Training curves** (Keras run → **Model metrics**, `epoch.loss` and `epoch.val_loss`): E1's
   curves are both still falling at epoch 3, so it is under-trained (fine for a small baseline).
   `val_loss` rising while `loss` keeps falling would mean overfitting. Early stopping then keeps
   the best epoch.
6. **Lineage tags:** `data_version` (`658e29cded93`), `git_commit`, `git_dirty`, and
   `airflow_dag_run_id` for runs made by Airflow. Only `git_dirty=false` runs count as evidence.
7. **Artifacts** of a Keras run:
   - `model/`: `model.keras`, `preprocessing.json`, `model_card.json`
   - `eval/`: Markdown report, per-label CSV, full metrics JSON
   - `data/manifest.json`: exactly which data the run used
8. **Reproducibility:** repeated E1 runs on the same machine have identical metrics (seeded).

---

## 8. Try the model on your own text: `redact`

```bash
make redact TEXT="Hi, I am Joseph Zarecki. Text me at 727-360-0442 or email josephzarecki55@yahoo.com"
scripts
un.cmd redact -Text "Hi, I am Joseph Zarecki. Text me at 727-360-0442 or email josephzarecki55@yahoo.com"
```

```text
Hi, I am [FIRST_NAME] [LAST_NAME]. Text me at [PHONE_NUMBER] or email [EMAIL]

model e2_dilated_cnn-9f3e9df7, 4 spans, 262.0 ms
  first_name                   1.00  Joseph
  last_name                    1.00  Zarecki
  phone_number                 1.00  727-360-0442
  email                        1.00  josephzarecki55@yahoo.com
```

Each span shows its label, the model's confidence, and the original text. A French-Canadian
prompt shows the locale risk from the proposal. The SIN and the postal code are redacted, but
"Bonjour" is flagged as a last name (low confidence, 0.41) and the postal code gets the wrong
type:

```text
> scripts
un.cmd redact -Model e2 -WordSnap -Text "Bonjour, je m'appelle Marie Tremblay, NAS 046 454 286, code postal K1N 6N5."
[LAST_NAME], je m'appelle [FIRST_NAME] [LAST_NAME], [NATIONAL_ID], code postal [LICENSE_PLATE].
```

Options:

| macOS / Linux | Windows | Effect |
| --- | --- | --- |
| `MODEL=e1` | `-Model e1` | which model: an experiment name/prefix, a `model_version`, or a model folder (default: the newest model) |
| `WORD_SNAP=1` | `-WordSnap` | whole-word redaction |

More options (threshold, reading a file or stdin, full JSON output) via
`uv run python -m src.serving.redactor --help`. The JSON is the **prediction contract** the browser
app will follow: `redacted_text`, `spans` (`start`, `end`, `label`, `score`), `model_version`,
`latency_ms`. Invalid input (empty text, > 20,000 characters, not a string) is rejected with an error code.

---

## 9. Tests and lint: `test`, `lint`

`test` runs unit tests and end-to-end smoke tests on tiny synthetic data (no download needed).
`lint` runs ruff lint and the format check, exactly like CI. Expected: all pass. On Windows the
Airflow DAG test is **skipped** (Airflow can't load there). CI runs it on Linux.

---

## 10. Airflow in Docker: `airflow-docker-up`, `airflow-docker-run`

Runs the same pipeline as a DAG, orchestrated by Airflow:
`ingest → validate_raw → preprocess → validate_processed → {evaluate_baseline, train}`.

```bash
# 1. Start Airflow (Docker Desktop must be running). First build ~5 min, then seconds.
make airflow-docker-up                 # Windows: scripts\run.cmd airflow-docker-up

# 2. Trigger a run and follow it until it finishes (prints each task's state).
make airflow-docker-run CONFIG=configs/experiments/e1_small_cnn_subset.yaml
scripts\run.cmd airflow-docker-run -Config configs/experiments/e1_small_cnn_subset.yaml

# 3. Stop it when done (state is kept for next time)
make airflow-docker-down               # Windows: scripts\run.cmd airflow-docker-down
```

| URL | What |
| --- | --- |
| <http://localhost:8080> | Airflow UI. **No login** (local use only) |
| <http://localhost:5001> | MLflow for runs made **inside the container** (experiment `pii-redaction-docker`) |

You can also trigger runs from the UI: open `pii_redaction_pipeline` → **Trigger** → set `config`
to an experiment config path. Without a config, the DAG trains the slow default model.

### What to look at in the Airflow UI

- **Graph view** of a run: data validation gates everything. `evaluate_baseline` and `train` then
  run in parallel, since both only need validated data.
- **A task → Logs:** `validate_raw` shows the validation summary, and `train` shows the per-epoch
  losses and final metrics.
- **Task details → retries:** `ingest` retries 3× with backoff (network), `train` retries once
  (transient failures), and validation tasks never retry (a failed data check needs a human).
- **XCom** of a task: the small JSON summary each step returns (e.g. `train` → `model_version`,
  metrics).
- **MLflow at :5001:** the run's `airflow_dag_run_id` tag links it back to the Airflow run.

### How it is set up

- `docker-compose.yml` + `docker/airflow.Dockerfile`: dependencies are baked into the image. The
  repository is **mounted** at `/app`, so code/DAG/config edits apply without a rebuild. Rebuild
  (`airflow-docker-up` does it) only when `uv.lock` changes.
- `configs/docker_overrides.yaml` (via `PII_CONFIG_OVERRIDES`) sends the container's MLflow runs,
  models and reports to the `airflow-state` Docker volume. This keeps your local `mlflow.db` and
  the committed `reports/` untouched. Data (`data/`) is shared with your machine.
- Reset everything Airflow-related: `docker compose down -v` (deletes the volume: Airflow history,
  container MLflow runs and models).

### Native Airflow (macOS / Linux only)

`make airflow` (UI; login printed in the terminal) and `make airflow-run CONFIG=...` (one
in-process run) also work without Docker. They store state in `.airflow/` and log to your local
`mlflow.db`.

---

## 11. Pipeline Docker image: `docker-build`, `docker-train`

The reproducible training environment as an image (no Airflow):

```bash
make docker-build                      # Windows: scripts\run.cmd docker-build
make docker-train CONFIG=configs/experiments/e1_small_cnn_subset.yaml
scripts\run.cmd docker-train -Config configs/experiments/e1_small_cnn_subset.yaml
```

- `docker-build` records the current git commit in the image (it has no `.git`), so MLflow runs
  made inside it still carry `git_commit`.
- `docker-train` mounts `data/`, `reports/` and `artifacts/`. The MLflow run stays inside the
  (deleted) container. Use the Airflow setup when you want to keep container runs.
- The image's default command (`pipelines.cli all`) runs every step with the slow default config.

---

## 12. GPU training: `train-gpu`

Trains on an NVIDIA GPU, with the same code, configs and outputs as `train`:

```bash
make train-gpu CONFIG=configs/experiments/e2_dilated_cnn.yaml
scripts
un.cmd train-gpu -Config configs/experiments/e2_dilated_cnn.yaml
```

> **Current status: it works, but it isn't much faster yet.** On an RTX 5090, E2 runs at ~116 ms
> per training step vs ~170 ms on a 16-core CPU: about **7.6 vs 11 min per epoch** (~1.5×). The
> E2 model is small and uses batches of 64, so the GPU spends much of its time waiting. Speeding
> this up (larger batches, mixed precision, input pipeline) is still to investigate. Until then
> the CPU is fine for E1–E4, and `train-gpu` mainly frees your CPU while training.
>
> Benchmark notes for whoever picks this up: XLA compilation matters (it's on by default; turning
> it off makes steps ~1.8× slower), and the progress/backup callbacks cost only ~4%.

**Requirements:** an NVIDIA GPU with a recent driver, and Docker Desktop. CUDA and cuDNN come
inside the image (locked versions, dependency group `gpu`). Nothing else to install. There is no
GPU path on macOS, where training runs on the CPU.

**Why Docker:** TensorFlow dropped native Windows GPU support after version 2.10. On Windows, GPU
training runs in a Linux container through Docker Desktop's GPU passthrough
(`docker/gpu.Dockerfile`, compose service `trainer-gpu`).

What's the same, and what differs, from `train`:

- Same: models go to `artifacts/models/`, reports to `reports/experiments/`, and `status`,
  `compare`, `redact`, `decode-sweep` and pause/resume all work.
- The MLflow run goes to the **Docker MLflow store** (<http://localhost:5001> while Airflow is up,
  experiment `pii-redaction-docker`): a container can't share the host's SQLite store. The
  `tracking_uri` in each summary JSON says which store a run is in.
- The run is tagged `hardware=gpu:<name>`. GPU and CPU results differ slightly (different
  floating-point kernels, see section 13), so compare experiments trained on the same hardware.

**First run on a new GPU generation:** TensorFlow 2.21 ships no precompiled kernels for the newest
cards (e.g. RTX 50xx, compute capability 12.0). CUDA compiles them on first use (a minute or two
of apparent idling, plus a warning saying it "could take 30 minutes"). The result is cached in the
`cuda-cache` Docker volume, so later runs start immediately.

---

## 13. Reproducibility notes (for the report)

| Observation | Explanation |
| --- | --- |
| Data validation reports are identical across Windows, macOS and Linux | Pinned dataset revision + deterministic preprocessing + `data_version` hashes source files with line endings normalised |
| Repeated E1 runs **on the same platform** give identical metrics (0.5446) | Seeds (`keras.utils.set_random_seed`) + seeded tf.data shuffling |
| A resumed run differs slightly from an uninterrupted one (E1: 0.5472 vs 0.5446) | The shuffle order restarts after a resume. Evidence runs should run uninterrupted, or be labelled `resumed=true` |
| E1 in Linux/Docker: recall 0.5445, macro-F1 0.596 vs Windows 0.5446 / 0.584 | TensorFlow's CPU kernels (oneDNN) differ between OS builds, so floating-point results differ slightly. Document this as a source of nondeterminism: compare experiments on one platform |
| Linux writes reports with LF endings into a Windows checkout | Content identical. `git status` shows them as modified on Windows only. Discard with `git checkout reports/` |

---

## 14. Where everything goes

| Path | Content | In Git? |
| --- | --- | --- |
| `data/raw/`, `data/processed/` | dataset + clean splits | no (reproduce with `data`) |
| `reports/validation/`, `reports/baseline/`, `reports/experiments/` | evidence for the report | **yes** |
| `artifacts/models/<model_version>/` | trained models | no (also in MLflow) |
| `artifacts/checkpoints/` | backups of unfinished training runs (removed when a run completes) | no |
| `mlflow.db`, `mlartifacts/` | local MLflow runs | no |
| Docker volume `airflow-state` | Airflow DB/logs, container MLflow runs, models, reports | no |
| `.airflow/` | native Airflow state (macOS/Linux) | no |

---

## 15. Troubleshooting

| Problem | Fix |
| --- | --- |
| `processed data is version X but ... give Y; re-run preprocess` | Data, split config or data code changed. Run `data` again. |
| `... not found; run the preprocess step first` | Run `data` first. |
| `raw file ... does not match` the manifest | `data/raw/` was modified. Delete it and run `data`. |
| `docker: ... cannot find the file specified` / `failed to connect to the docker API` | Docker Desktop isn't running. Start it and wait until it says "Engine running". |
| `port is already allocated` (8080 / 5001) | Another app uses the port. Stop it, or change the left side of `ports:` in `docker-compose.yml`. |
| Airflow UI doesn't load right after `airflow-docker-up` | It needs ~30 s. `airflow-docker-logs` shows progress. |
| `No module named 'fcntl'` | Native Airflow on Windows. Use `airflow-docker-*` instead. |
| Lots of TensorFlow `oneDNN` / CUDA messages | Harmless. (`cuInit` errors in Docker just mean "no GPU".) |
| `make: command not found` (Windows) | Use `scripts\run.cmd <task>`. |
| A step crashed half-way | Steps are re-runnable. Fix the cause and re-run the same task. `train` resumes from its last completed epoch (section 6). |
| Commit fails with `error: unknown option 'deduplicate'` | Git is older than 2.31, which the pre-commit hooks need. Upgrade Git (section 2). |
| Training resumes when you wanted a fresh start | Delete `artifacts/checkpoints/<experiment>-<hash>/` (or set `train.resume: false`). |
