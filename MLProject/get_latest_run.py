from __future__ import annotations

from pathlib import Path
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException


EXPERIMENT_NAME = "Latihan Credit Scoring CI"
RUN_ID_PATH = Path("latest_run_id.txt")


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def maybe_init_dagshub() -> None:
    if os.getenv("DAGSHUB_MLFLOW", "").lower() not in {"1", "true", "yes"}:
        mlflow.set_tracking_uri("file:./mlruns")
        return

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if not repo_owner or not repo_name:
        raise ValueError("DAGSHUB_REPO_OWNER dan DAGSHUB_REPO_NAME wajib diisi.")

    import dagshub

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def get_saved_run_id() -> str | None:
    if not RUN_ID_PATH.exists():
        return None

    run_id = RUN_ID_PATH.read_text(encoding="utf-8").strip()
    if not run_id:
        return None

    try:
        mlflow.get_run(run_id)
    except MlflowException:
        return None

    return run_id


def get_experiment_ids() -> list[str]:
    experiment_ids: list[str] = []
    for experiment_name in (EXPERIMENT_NAME, "Default"):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None and experiment.experiment_id not in experiment_ids:
            experiment_ids.append(experiment.experiment_id)

    if experiment_ids:
        return experiment_ids

    return [experiment.experiment_id for experiment in mlflow.search_experiments()]


def main() -> None:
    maybe_init_dagshub()
    saved_run_id = get_saved_run_id()
    if saved_run_id is not None:
        RUN_ID_PATH.write_text(saved_run_id, encoding="utf-8")
        return

    experiment_ids = get_experiment_ids()
    if not experiment_ids:
        raise RuntimeError("Experiment tidak ditemukan.")

    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("Belum ada MLflow run.")

    run_id = runs.iloc[0]["run_id"]
    RUN_ID_PATH.write_text(str(run_id), encoding="utf-8")


if __name__ == "__main__":
    main()
