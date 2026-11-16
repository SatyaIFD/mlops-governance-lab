import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from src.features.velocity import CANONICAL_AML_FEATURES, StreamingStateManager
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.models.anomaly_detector import AMLAnomalyDetector, TRUSTED_SKOPS_TYPES
from src.streaming.generator import transaction_stream_generator


def select_empirical_threshold(y_val: pd.Series, val_probs: np.ndarray) -> float:
    """Selects the probability threshold maximizing F1 on validation data subject to recall >= 0.25 (falling back to unconstrained F1)."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    if len(thresholds) == 0:
        return 0.50
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-9
    )
    valid_mask = recalls[:-1] >= 0.25
    if np.any(valid_mask):
        masked_f1 = np.where(valid_mask, f1_scores, -1.0)
        best_idx = int(np.argmax(masked_f1))
    else:
        best_idx = int(np.argmax(f1_scores))
    raw_cutoff = float(thresholds[best_idx])
    return round(max(0.25, min(0.85, raw_cutoff)), 2)


def prepare_training_data(
    csv_path: Path, warmup_rows: int = 50000, train_rows: int = 50000
) -> pd.DataFrame:
    """Streams warmup_rows + train_rows through StreamingStateManager and returns the enriched training slice."""
    total_rows = warmup_rows + train_rows
    print(
        f"🧠 Building historical 7-feature state matrix ({warmup_rows} warmup + {train_rows} training rows)..."
    )
    tmp_dlq = csv_path.parent / ".train_dlq_tmp"
    pseudonymize = os.getenv("AML_PSEUDONYMIZE_ACCOUNTS", "1") == "1"
    ingestor = StreamingIngestionEngine(
        dlq_output_dir=tmp_dlq, pseudonymize_accounts=pseudonymize
    )
    state_manager = StreamingStateManager()
    enriched_records = []
    try:
        for idx, raw_tx in enumerate(
            transaction_stream_generator(csv_path, delay_seconds=0.0)
        ):
            if idx >= total_rows:
                break
            is_valid, clean_tx = ingestor.validate_and_route(raw_tx)
            if not is_valid or clean_tx is None:
                continue
            enriched = state_manager.update_and_enrich(clean_tx)
            if idx >= warmup_rows:
                if (
                    "Is_laundering" not in raw_tx
                    or raw_tx["Is_laundering"] is None
                    or str(raw_tx["Is_laundering"]).strip() == ""
                ):
                    raise ValueError(
                        "Missing required 'Is_laundering' target label in training stream."
                    )
                label_val = int(raw_tx["Is_laundering"])
                if label_val not in (0, 1):
                    raise ValueError(
                        f"Invalid 'Is_laundering' label value: {label_val}"
                    )
                row = {col: enriched[col] for col in CANONICAL_AML_FEATURES}
                row["Is_laundering"] = label_val
                enriched_records.append(row)
    finally:
        shutil.rmtree(tmp_dlq, ignore_errors=True)
    df = pd.DataFrame(enriched_records)
    if df.empty or df["Is_laundering"].nunique() < 2:
        raise ValueError(
            "Prepared training data must contain both classes (0 and 1) in 'Is_laundering'."
        )
    return df


def execute_training_pipeline():
    root_dir = Path(__file__).resolve().parents[2]
    data_path = Path(os.getenv("AML_DATA_PATH", str(root_dir / "data" / "SAML-D.csv")))
    artifacts_dir = root_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    skops_artifact_path = artifacts_dir / "aml_streaming_model"

    db_path = root_dir.parent / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_tracking_uri(tracking_uri)

    mlruns_dir = root_dir.parent / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    exp_name = "aml_streaming_governance"
    client = MlflowClient()
    if client.get_experiment_by_name(exp_name) is None:
        client.create_experiment(exp_name, artifact_location=mlruns_dir.as_uri())
    mlflow.set_experiment(exp_name)

    warmup_rows = 50000
    train_rows = 50000
    df = prepare_training_data(
        data_path, warmup_rows=warmup_rows, train_rows=train_rows
    )
    split_idx = int(len(df) * 0.80)
    train_split = df.iloc[:split_idx].reset_index(drop=True)
    val_split = df.iloc[split_idx:].reset_index(drop=True)

    X_train = train_split[CANONICAL_AML_FEATURES]
    X_val = val_split[CANONICAL_AML_FEATURES]
    y_val = val_split["Is_laundering"]

    with mlflow.start_run(run_name="stateful_rf_7features") as run:
        detector = AMLAnomalyDetector()
        detector.train(train_split, training_window_end_row=warmup_rows + train_rows)

        val_probs = detector.model.predict_proba(X_val)[:, 1]
        empirical_cutoff = select_empirical_threshold(y_val, val_probs)
        detector.threshold = empirical_cutoff
        val_preds = (val_probs >= empirical_cutoff).astype(int)

        val_auc = (
            float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5
        )
        val_f1 = float(f1_score(y_val, val_preds, zero_division=0))

        rf_params = detector.model.get_params()
        mlflow.log_params(
            {
                "model_type": type(detector.model).__name__,
                "n_estimators": rf_params["n_estimators"],
                "max_depth": rf_params["max_depth"],
                "class_weight": str(rf_params["class_weight"]),
                "features": ",".join(CANONICAL_AML_FEATURES),
                "calibrated_threshold": empirical_cutoff,
                "training_window_end_row": detector.training_window_end_row,
            }
        )
        mlflow.log_metrics(
            {
                "val_roc_auc": round(val_auc, 4),
                "val_f1_score": round(val_f1, 4),
            }
        )

        signature = infer_signature(
            X_train, detector.model.predict_proba(X_train)[:, 1]
        )
        mlflow.sklearn.log_model(
            sk_model=detector.model,
            artifact_path="aml_model",
            registered_model_name="aml_streaming_model",
            signature=signature,
            skops_trusted_types=TRUSTED_SKOPS_TYPES,
        )
        detector.save(skops_artifact_path, mlflow_run_id=run.info.run_id)
        mlflow.log_artifact(str(skops_artifact_path), artifact_path="skops_bundle")
        mlflow.log_artifact(
            str(skops_artifact_path.with_suffix(".meta.json")),
            artifact_path="skops_bundle",
        )
        print(
            f"💾 7-feature AML model registered to MLflow and saved to {skops_artifact_path} "
            f"(val_auc={val_auc:.4f}, threshold={empirical_cutoff}, window_end={detector.training_window_end_row})."
        )


if __name__ == "__main__":
    execute_training_pipeline()
