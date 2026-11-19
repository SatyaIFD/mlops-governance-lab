
# ⚖️ Model Risk Assessment & Algorithmic Mitigation Protocol

**Project 2: Loan Default Risk Underwriting Engine**

---

## 1. Core Model Failure Modes & Operational Risk

This model scores retail loan applications to assist human credit underwriting. Unmitigated failures introduce two primary risks:

* **False Negatives (Under-prediction of Default Risk):** Recommending approval for high-risk applicants, increasing Non-Performing Loan (NPL) ratios.
* **False Positives & Demographic Disparity:** Systematically recommending rejection for younger applicants (`Age < 30`) due to shorter credit histories.

## 2. Quantitative Guardrails Implemented in Code

The pipeline enforces automated checks across evaluation (`src/evaluation/evaluate.py`), drift validation (`src/validation/drift_check.py`), and live telemetry auditing (`src/monitoring/drift_monitor.py`):

| Check Layer | Metric / Indicator | Implemented Threshold | Pipeline Behavior in Code |
| --- | --- | --- | --- |
| **Holdout Performance** | Test ROC-AUC (`evaluate.py`) | $\text{ROC-AUC} \ge 0.70$ | Sets `minimum_performance_passed = True`; exits non-zero if breached. |
| **Holdout Fairness (`evaluate.py`)** | DI 95% CI, Equal Opportunity TPR 95% CI, FPR gap, & Reg B `Age >= 62` CI | DI & TPR 95% CI $\subseteq [0.80, 1.25]$, $\text{FPR gap} \le 0.05$, Elderly `62+` CI lower $\ge 1.0$ | Evaluated out-of-sample on `test.parquet` at the validation-calibrated unified threshold (`0.1358`); exits non-zero if any gate fails. |
| **Telemetry Fairness (`drift_monitor.py`)** | Disparate Impact Ratio & 95% Katz CI (`UNDER_30` vs. `30_AND_OVER`) | $0.80 \le \text{DI}_{95\%\text{ CI}} \le 1.25$ (`min_cohort_size = 30`) | Reports `COMPLIANT` when the full 95% CI is within `[0.80, 1.25]`, `NON-COMPLIANT (BREACH)` when entirely outside (with Haldane-Anscombe `+0.5` correction), `INCONCLUSIVE (CI_STRADDLES_THRESHOLD)` when crossing a boundary, or `INCONCLUSIVE (INSUFFICIENT_DATA)` when below `min_cohort_size = 30` or baseline approvals are zero. Live and synthetic traffic are tracked separately. |
| **Population Drift** | Population Stability Index (`drift_check.py`) | $\text{PSI} < 0.10$ (`STABLE`), $0.10 \le \text{PSI} < 0.25$ (`MODERATE_SHIFT`), $\text{PSI} \ge 0.25$ (`SEVERE_DRIFT`) | Evaluates 15 features (8 numerical + 7 categorical) on `--evaluate-holdout`, and 4 core numerical features (`CreditScore`, `Income`, `LoanAmount`, `DTIRatio`) on telemetry streams (`min_telemetry_rows = 500`); exits non-zero when `--fail-on-drift` or `--simulate-drift` detects severe drift. |
| **Operational Resilience** | Consecutive Inference Exceptions (`dora_circuit_breaker.py`) | $\ge 3$ failures (`recovery_timeout_s = 30.0`) | Trips `MLSystemCircuitBreaker` to `OPEN`, returning `HTTP 503` with single-probe `HALF_OPEN` recovery locking. |

## 3. Implemented Fair Lending & Resilience Controls

* **Pre-Processing Sample Reweighting (`src/training/train.py`):** `calculate_mitigation_weights()` computes inverse joint-probability weights across age cohorts (`Age < 30` vs. `Age >= 30`) and default labels (`Default == 0` vs. `Default == 1`) prior to fitting `XGBClassifier`.
* **Single-Row Categorical Preservation (`src/inference/main.py`):** `prepare_single_row_features()` encodes incoming requests with `drop_first=False` before reindexing to the model's exact training feature columns so single-row payloads never lose their active category level.
* **Mandatory Human-in-the-Loop Routing (`src/inference/main.py`):** Every `/predict` call invokes `enforce_human_in_the_loop()`, setting `action: "MANUAL_REVIEW_REQUIRED"`, `review_status: "PENDING_HUMAN_REVIEW"`, and `final_decision: null`.
