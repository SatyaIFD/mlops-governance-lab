"""
Model Training & Orchestration Module - Project 1: Credit Card Fraud Detection
Fits preprocessing strictly on the training split, splits train into 80% fit and 20% validation,
selects scale_pos_weight on the validation split (maximizing validation PR-AUC subject to validation
precision >= 0.95), and logs validation and holdout metrics to MLflow.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.utils import (
    CANONICAL_FEATURE_COLS,
    save_scaler_params,
    split_and_scale_without_leakage,
)
from src.training.data_splitter import prepare_stratified_splits


def _select_validation_model(
    X_fit: pd.DataFrame, y_fit: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
):
    """Selects scale_pos_weight from (1.0, 2.0, 4.0) on the 20% validation split maximizing val PR-AUC subject to val precision >= 0.95."""
    best_model = None
    best_params = None
    best_val_metrics = None
    best_score = -1.0
    for cand_spw in (1.0, 2.0, 4.0):
        cand_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": cand_spw,
            "random_state": 42,
        }
        clf = xgb.XGBClassifier(**cand_params)
        clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)
        val_prec = float(precision_score(y_val, val_preds, zero_division=0))
        val_rec = float(recall_score(y_val, val_preds, zero_division=0))
        val_f1 = float(f1_score(y_val, val_preds, zero_division=0))
        val_pr_auc = float(average_precision_score(y_val, val_probs))
        val_roc_auc = float(roc_auc_score(y_val, val_probs))
        qualifies = val_prec >= 0.95
        rank_score = val_pr_auc if qualifies else (val_pr_auc * 0.5)
        if rank_score > best_score:
            best_score = rank_score
            best_model = clf
            best_params = cand_params
            best_val_metrics = {
                "val_roc_auc": round(val_roc_auc, 4),
                "val_pr_auc": round(val_pr_auc, 4),
                "val_f1": round(val_f1, 4),
                "val_precision": round(val_prec, 4),
                "val_recall": round(val_rec, 4),
            }
    return best_model, best_params, best_val_metrics


def execute_training_pipeline(processed_data_file: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    data_path = Path(processed_data_file).resolve()
    project_root = (
        data_path.parents[1] if "data" in data_path.parts else data_path.parent
    )
    raw_csv_path = project_root / "data" / "creditcard.csv"
    scaler_artifact_path = project_root / "artifacts" / "scaler_params.json"
    processed_dir = project_root / "data" / "processed"

    mlruns_dir = project_root.parent / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = "credit_card_fraud_governance"
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name, artifact_location=mlruns_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    if raw_csv_path.exists():
        print(
            f"🔒 Splitting raw dataset ({raw_csv_path.name}) prior to scaling to prevent test leakage..."
        )
        raw_df = pd.read_csv(raw_csv_path)
        train_df, test_df, scaler_params = split_and_scale_without_leakage(raw_df)
        save_scaler_params(scaler_params, scaler_artifact_path)
        processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(processed_dir / "train.parquet", index=False)
        test_df.to_parquet(processed_dir / "test.parquet", index=False)
        X_train_full = train_df[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        y_train_full = train_df["Class"].reset_index(drop=True)
        X_test = test_df[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        y_test = test_df["Class"].reset_index(drop=True)
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"🚨 FATAL: Processed data missing at {data_path}.")
        df = pd.read_parquet(data_path)
        X_train_full, X_test, y_train_full, y_test = prepare_stratified_splits(df)
        X_train_full = X_train_full[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        X_test = X_test[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        y_train_full = y_train_full.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    with mlflow.start_run() as run:
        print(
            f"🚀 Selecting hyperparameters on validation split inside run: {run.info.run_id}"
        )
        model, params, val_metrics = _select_validation_model(
            X_fit, y_fit, X_val, y_val
        )
        mlflow.log_params(params)
        for mk, mv in val_metrics.items():
            mlflow.log_metric(mk, mv)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)
        accuracy = float((preds == y_test).mean())
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))

        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("roc_auc", round(roc_auc, 4))
        mlflow.log_metric("pr_auc", round(pr_auc, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        print(
            f"📊 Selected scale_pos_weight={params['scale_pos_weight']} (val PR-AUC={val_metrics['val_pr_auc']:.4f}, val Prec={val_metrics['val_precision']:.4f}) | "
            f"Holdout: ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}"
        )

        if scaler_artifact_path.exists():
            mlflow.log_artifact(
                str(scaler_artifact_path), artifact_path="preprocessing"
            )

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="fraud_detection_xgb",
        )
        print("🟢 Model binaries successfully logged and registered.")


if __name__ == "__main__":
    execute_training_pipeline(
        processed_data_file=str(BASE_DIR / "data/processed_features.parquet"),
        tracking_uri=f"sqlite:///{BASE_DIR.parent}/mlflow.db",
    )
