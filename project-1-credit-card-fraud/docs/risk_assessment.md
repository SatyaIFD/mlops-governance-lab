# Algorithmic Risk Assessment & Operational Segment Telemetry Report

## Executive Summary
This report documents the out-of-sample quantitative validation for the Credit Card Fraud Detection Classifier (`models:/fraud_detection_xgb/11`) on `data/processed/test.parquet` (`56,746` transactions, `95` frauds). The model achieves **`ROC-AUC = 0.9603`** (gate `>= 0.90`) and **`PR-AUC = 0.7909`** (gate `>= 0.75`), passing the enforced discrimination gate (`performance_gate_passed: true`). At the served `0.50` probability threshold, diagnostic overall recall is **`75.79%`** (95% Wilson CI `[0.6628, 0.8329]`) with **`94.74%` precision** (`73.08%` recall on high-value transactions and `76.81%` recall on low-value transactions). The informational flag-rate ratio between high-value ($\ge 80\text{th}$ percentile) and low-value ($< 80\text{th}$ percentile) segments is **`1.656`** (95% CI `[1.0093, 2.7179]`, `within_80_125_band: false`), reflecting the higher empirical fraud prevalence in large transactions (`0.232%` vs. `0.152%`).

---

## 1. Interpretability & Offline Explainability Analysis (XAI)
* **Mechanism:** Offline game-theoretic marginal contributions analyzed via `SHAP` (`TreeExplainer`).
* **Findings:** Model decision boundaries are primarily driven by PCA-transformed latent transaction components (**`V14`**, **`V17`**, and **`V12`**), which are monitored for Population Stability Index (PSI) drift alongside `scaled_amount` and `scaled_time` in `src/validation/drift_check.py`.

---

## 2. Operational Transaction Segment Telemetry (Informational)
Because the public ULB credit card dataset contains only PCA-anonymized features (`V1`–`V28`), `Time`, and `Amount` (with zero demographic attributes), transaction amount tiers are not a legal protected class. `src/governance/audit.py` gates `--enforce-gate` on threshold-independent discrimination (`ROC-AUC >= 0.90` and `PR-AUC >= 0.75`) to avoid discrete single-transaction fragility on the 95 holdout positives, while recording operating-point recall/precision and segment flag-rate ratios as diagnostic telemetry:

* **Low-Value Segment ($< 80\text{th}$ percentile):** Flag Rate = **`0.001186`**, Recall = **`0.7681`** (95% CI `[0.656, 0.8519]`), Precision = **`0.9815`**
* **High-Value Segment ($\ge 80\text{th}$ percentile):** Flag Rate = **`0.001964`**, Recall = **`0.7308`** (95% CI `[0.5392, 0.863]`), Precision = **`0.8636`**
* **Informational Flag-Rate Ratio:** **`1.656`** (`within_80_125_band: false`, `performance_gate_passed: true`)
