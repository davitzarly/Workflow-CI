from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "is_canceled"
RANDOM_STATE = 42
EXPERIMENT_NAME = "Latihan Credit Scoring CI"
RUN_NAME = "ci_random_forest"


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


def load_data() -> pd.DataFrame:
    candidates = [
        Path("hotel_bookings.csv"),
        Path("../hotel_bookings.csv"),
        Path("../../hotel_bookings.csv"),
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, na_values=["NULL", "NA", ""])

    rng = np.random.default_rng(RANDOM_STATE)
    rows = 500
    return pd.DataFrame(
        {
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], rows),
            "lead_time": rng.integers(0, 365, rows),
            "stays_in_weekend_nights": rng.integers(0, 4, rows),
            "stays_in_week_nights": rng.integers(1, 10, rows),
            "adults": rng.integers(1, 4, rows),
            "children": rng.integers(0, 3, rows),
            "babies": rng.integers(0, 2, rows),
            "market_segment": rng.choice(["Online TA", "Direct", "Corporate"], rows),
            "deposit_type": rng.choice(["No Deposit", "Non Refund"], rows),
            "customer_type": rng.choice(["Transient", "Contract"], rows),
            "adr": rng.normal(95, 35, rows).clip(0, 500),
            TARGET_COLUMN: rng.integers(0, 2, rows),
        }
    )


def find_preprocessed_file(name: str) -> Path | None:
    data_dir = Path("namadataset_preprocessing")
    for suffix in (".csv", ".csv.gz", ".csv.zip"):
        path = data_dir / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    train_path = find_preprocessed_file("train_preprocessed")
    test_path = find_preprocessed_file("test_preprocessed")
    if train_path is None or test_path is None:
        return None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    return x_train, x_test, y_train, y_test


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    df = df.copy()
    df = df.drop(columns=["reservation_status", "reservation_status_date"], errors="ignore")
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )

    return x_train, x_test, y_train, y_test, preprocessor


def main() -> None:
    maybe_init_dagshub()
    project_run_id = os.getenv("MLFLOW_RUN_ID")
    if project_run_id is None:
        mlflow.set_experiment(EXPERIMENT_NAME)

    preprocessed = load_preprocessed_data()
    if preprocessed is None:
        x_train, x_test, y_train, y_test, preprocessor = prepare_data(load_data())
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        mlflow_input_example = x_test.head(5)
    else:
        x_train, x_test, y_train, y_test = preprocessed
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
        mlflow_input_example = x_test.head(5)

    with mlflow.start_run(run_id=project_run_id, run_name=RUN_NAME) as run:
        mlflow.set_tag("mlflow.runName", RUN_NAME)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "random_state": RANDOM_STATE,
                "data_source": "namadataset_preprocessing" if preprocessed is not None else "raw_or_synthetic",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=mlflow_input_example)

        Path("metric_info.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        Path("latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        mlflow.log_artifact("metric_info.json")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
