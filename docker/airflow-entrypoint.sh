#!/bin/sh
# Start the MLflow UI for container runs (port 5001) and Airflow standalone (port 8080).
set -e
mkdir -p /state/mlflow /state/airflow
airflow db migrate > /state/airflow/db-migrate.log 2>&1
mlflow ui --backend-store-uri sqlite:////state/mlflow/mlflow.db \
    --host 0.0.0.0 --port 5001 > /state/mlflow-ui.log 2>&1 &
exec airflow standalone
