#!/bin/sh
# Trigger one run of the pipeline DAG through the Airflow scheduler and wait for it to finish.
# Usage (from the host, with the airflow service running):
#   docker compose exec airflow airflow-run configs/experiments/e1_small_cnn_subset.yaml
# Watch it live in the UI at http://localhost:8080 (Graph view). Exit code 0 = run succeeded.
set -e
DAG=pii_redaction_pipeline
CONFIG="${1:-configs/default.yaml}"
RUN_ID="cli__$(date -u +%Y-%m-%dT%H-%M-%S)"

airflow dags reserialize > /dev/null 2>&1
airflow dags unpause "$DAG" > /dev/null 2>&1
airflow dags trigger "$DAG" --run-id "$RUN_ID" --conf "{\"config\": \"$CONFIG\"}" > /dev/null
echo "Triggered $DAG run $RUN_ID (config: $CONFIG)"
echo "Follow it at http://localhost:8080/dags/$DAG"

# Poll the REST API (the CLI mixes log lines into its output). Prints each task state change.
exec python - "$DAG" "$RUN_ID" <<'EOF'
import json, sys, time, urllib.request

dag, run_id = sys.argv[1], sys.argv[2]
base = f"http://localhost:8080/api/v2/dags/{dag}/dagRuns/{run_id}"

def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)

seen, state = {}, "queued"
while state not in ("success", "failed"):
    time.sleep(5)
    try:
        state = get(base)["state"]
        tasks = get(base + "/taskInstances")["task_instances"]
    except OSError as exc:  # API briefly unavailable: keep waiting
        print(f"(waiting for the API: {exc})")
        continue
    for t in sorted(tasks, key=lambda t: t["start_date"] or "~"):
        if t["state"] and seen.get(t["task_id"]) != t["state"]:
            seen[t["task_id"]] = t["state"]
            print(f"{time.strftime('%H:%M:%S')}  {t['task_id']:<20} {t['state']}", flush=True)
print(f"Run {run_id}: {state}")
sys.exit(0 if state == "success" else 1)
EOF
