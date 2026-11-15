# Algorithmic Risk Assessment & Operational Segment Telemetry Report

## Executive Summary
This report documents the out-of-sample quantitative validation for the Credit Card Fraud Detection Classifier (`models:/fraud_detection_xgb/12`) on `data/processed/test.parquet` (`56,746` transactions, `95` frauds). Using `scale_pos_weight=1.0` (selected on the held-out 20% validation split by maximizing validation PR-AUC subject to validation precision `>= 0.95`), the model achieves **`ROC-AUC = 0.9659`** (gate `>= 0.90`) and **`PR-AUC = 0.8166`** (gate `>= 0.75`), and at the served `0.50` probability threshold achieves **`overall recall = 71.58%`** (gate `>= 60.0%`, 95% Wilson CI `[0.6181, 0.7967]`) and **`overall precision = 98.55%`** (gate `>= 80.0%`, `performance_gate_passed: true`). Diagnostic sub-slice recall is **`65.38%` (`94.44%` precision)** on high-value transactions and **`73.91%` (`100.00%` precision)** on low-value transactions. The informational flag-rate ratio between high-value ($\ge 80\text{th}$ percentile) and low-value ($< 80\text{th}$ percentile) segments is **`1.4348`** (95% CI `[0.8387, 2.4546]`, `within_80_125_band: false`, `ci_band_status: CI_STRADDLES_80_125_BAND`), reflecting the higher empirical fraud prevalence in large transactions (`0.232%` vs. `0.152%`).

---

## 1. Interpretability & Offline Explainability Analysis (XAI)
* **Mechanism:** Offline game-theoretic marginal contributions analyzed via `SHAP` (`TreeExplainer`).
* **Findings:** Model decision boundaries are primarily driven by PCA-transformed latent transaction components (**`V14`**, **`V17`**, and **`V12`**), which are monitored for Population Stability Index (PSI) drift alongside `scaled_amount` and `scaled_time` in `src/validation/drift_check.py`.

---

## 2. Operational Transaction Segment Telemetry (Informational)
Because the public ULB credit card dataset contains only PCA-anonymized features (`V1`–`V28`), `Time`, and `Amount` (with zero demographic attributes), transaction amount tiers are not a legal protected class. `src/governance/audit.py` gates `--enforce-gate` on both threshold-independent discrimination (`ROC-AUC >= 0.90`, `PR-AUC >= 0.75`) and deployed `0.50` operating-point business floors (`overall_recall >= 0.60`, `overall_precision >= 0.80`), while recording segment flag-rate ratios as diagnostic telemetry:

* **Low-Value Segment ($< 80\text{th}$ percentile):** Flag Rate = **`0.001120`**, Recall = **`0.7391`** (95% CI `[0.6249, 0.8281]`), Precision = **`1.0`**
* **High-Value Segment ($\ge 80\text{th}$ percentile):** Flag Rate = **`0.001607`**, Recall = **`0.6538`** (95% CI `[0.4622, 0.8059]`), Precision = **`0.9444`**
* **Informational Flag-Rate Ratio:** **`1.4348`** (`within_80_125_band: false`, `ci_band_status: CI_STRADDLES_80_125_BAND`, `performance_gate_passed: true`)
