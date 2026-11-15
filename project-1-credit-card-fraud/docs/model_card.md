# Model Card: Credit Card Fraud Detection Classifier (Version 1.0.0)

## 1. Model Details
* **Developer:** Enterprise Risk MLOps Team
* **Model Architecture:** XGBoost Classifier (`XGBClassifier`, `scale_pos_weight=4.0`, `max_delta_step=1`)
* **Engine Dependencies:** `xgboost==3.4.1`, `mlflow==3.18.0`, `python 3.13.14`
* **Registered Model URI:** `models:/fraud_detection_xgb/11` (SQLite Backend Tracking Registry)
* **Served Decision Threshold:** `0.50`
* **License:** Internal Financial Operations Use Only

## 2. Intended Use
* **Primary Scope:** Baseline model binary for validating end-to-end MLOps pipeline architecture, leakage-free train/test scaling, containerization, CI/CD automation, and DORA circuit-breaker resilience.
* **Out-of-Scope Uses:** Direct autonomous account termination without downstream human review.

## 3. Quantitative Holdout Performance & Business Gate (`src/governance/audit.py`)
* **Holdout ROC-AUC / PR-AUC (Enforced Gate):** `0.9603` (gate `>= 0.90`) / `0.7909` (gate `>= 0.75`)
* **Overall Holdout Recall / Precision (Diagnostic at `0.50`):** `0.7579` (95% Wilson CI: `[0.6628, 0.8329]`) / `0.9474`
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Diagnostic Recall / Precision:** `0.7308` (95% Wilson CI: `[0.5392, 0.863]`) / `0.8636` (`19/26` frauds detected)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Diagnostic Recall / Precision:** `0.7681` (95% Wilson CI: `[0.656, 0.8519]`) / `0.9815` (`53/69` frauds detected)
* **Data Drift Watchdog:** Custom Population Stability Index (PSI) tracker (`src/monitoring/observability.py`) with Laplace smoothing ($10^{-4}$).

## 4. Informational Operational Transaction Segment Telemetry
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Flag Rate:** `0.001964` (`22/11203`, fraud prevalence `0.232%`)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Flag Rate:** `0.001186` (`54/45543`, fraud prevalence `0.152%`)
* **Informational Segment Flag-Rate Ratio:** **`1.656`** (95% Katz CI: `[1.0093, 2.7179]`, `within_80_125_band: false` — Informational operational metric only; transaction amount tier is not a protected demographic attribute and higher-amount tiers exhibit higher base fraud prevalence (0.232% vs 0.152%).)
