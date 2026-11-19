# MLOps & AI Governance Lab

[![Python](https://img.shields.io/badge/Python-3.13.14--slim-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-hosted MLOps and AI governance learning lab for practicing end-to-end ML pipeline engineering across three financial-domain scenarios: batch fraud classification, fair-lending underwriting with human-in-the-loop (HITL) routing, and stateful streaming anomaly detection. The lab focuses on schema contracts, statistical drift and fairness auditing, automated CI/CD security scanning, and programmatic resilience controls.

## 🧪 Lab Architecture & Projects

This monorepo houses three self-contained pipelines sharing a local Nginx gateway, an isolated MLflow tracking registry, an Airflow 3.4 scheduler, and a shared `/compliance` package:

* **Project 1 — Credit Card Fraud (`project-1-credit-card-fraud/`):** Leakage-free train/test scaling, validation-selected XGBoost training (`scale_pos_weight=1.0`), holdout performance gating (`ROC-AUC`, `PR-AUC`, recall/precision with 95% Wilson CIs), informational transaction-tier telemetry, and PSI drift validation.
* **Project 2 — Loan Default Underwriting (`project-2-loan-default/`):** Sample-reweighted XGBoost with a validation-calibrated cohort-blind threshold (`0.1358`), single-row categorical alignment, mandatory `PENDING_HUMAN_REVIEW` HITL metadata, data-minimized telemetry (`age_bucket`, zero `MaritalStatus`), and Airflow-orchestrated fairness/drift auditing.
* **Project 3 — Stateful AML Streaming (`project-3-aml-streaming/`):** Headless consumer computing 7 rolling event-time graph and velocity features with 24-hour watermark eviction, Dead-Letter Queue (DLQ) isolation and replay (`scripts/replay_dlq.py`), key-versioned HMAC account pseudonymization, and `skops` SHA-256 model verification.

## 🔎 Scope & Known Design Boundaries

This repository is an engineering sandbox and learning lab, not a multi-region production banking platform:

* **Local Infrastructure:** Runs as a single-host Docker Compose fleet backed by SQLite (`./mlflow_db/mlflow.db` and `./airflow/db/airflow.db`) with Airflow `LocalExecutor` in a standalone container.
* **Honest Live vs. Synthetic Telemetry:** Because no external users send traffic to `/api/v1/loan/`, live 500-row drift and fairness gates remain honestly `NOT_EVALUATED` / `SKIPPED_INSUFFICIENT_TELEMETRY`, while synthetic holdout streams are isolated on loopback port `127.0.0.1:8081` (`synthetic_representative_simulation`).
* **Baseline Models on Public Datasets:** Metrics reflect reproducible baselines on public datasets (ULB Credit Card Fraud, Kaggle Loan Default, and SAML-D AML)—for example, Project 3's 7-feature classifier achieves `26.23%` out-of-sample recall and `0.48%` alert precision across 100,000 events (`122` laundering cases).
* **Container Vulnerability Scanning:** Trivy enforces blocking `CRITICAL`/`HIGH` gates (`exit-code: '1'`, `ignore-unfixed: true`) on all four owned images (Fraud, Loan, AML, MLflow Server), and runs report-only (`exit-code: '0'`) on the custom `apache/airflow:3.4.0` image.

## 🛡️ Automated CI/CD & DevSecOps

* **Linting & SAST:** Automated code formatting and lint checks via **Ruff** (`make lint`) and static security analysis via **Bandit** (`make security`).
* **Automated Test Suite (`make test`):** 75 unit, API contract, property-based (`hypothesis`), concurrency, failure-injection, and notebook regression tests across all three projects and `/compliance`.
* **Container Scanning & GHCR Publishing:** Service images are scanned with **Trivy** (`aquasecurity/trivy-action@v0.36.0`) and published to **GitHub Container Registry (GHCR)** after passing lint, SAST, test, and container gates.

## ⚖️ Governance & Resilience Utilities (`/compliance`)

The `/compliance` package implements engineering patterns inspired by GDPR, EU AI Act, and DORA requirements:

* **GDPR Erasure Utility (`gdpr_purge.py`):** Journaled, crash-recoverable erasure (`INITIATED` $\rightarrow$ `FILE_STAGED` $\rightarrow$ `FILE_SWAPPED` $\rightarrow$ `DB_COMMITTED` $\rightarrow$ `VERIFIED`) supporting Parquet, CSV, and active/rotated `.jsonl` / `.jsonl.*.gz` archives alongside scoped `user_id` deletions in SQLite with automatic `.bak` rollback on failure.
* **EU AI Act HITL Router (`eu_ai_act_hitl_router.py`):** Tags high-risk credit/fraud actions with `MANUAL_REVIEW_REQUIRED` and `PENDING_HUMAN_REVIEW` so downstream workflows require human adjudication (`final_decision: null`).
* **DORA Circuit Breaker (`dora_circuit_breaker.py`):** Thread-safe circuit breaker with exclusive `HALF_OPEN` single-probe locking that fails closed (`HTTP 503`) after consecutive inference exceptions.
* **Verification (`test_compliance.py`):** Property-based, concurrency, crash-recovery, compression-race, and notebook integrity tests.

## 🚀 Quickstart & Local Setup

### 1. Clone, Configure & Bootstrap

```bash
git clone https://github.com/SatyaIFD/mlops-governance-lab.git
cd mlops-governance-lab
[ -f .env.example ] && cp -n .env.example .env || true
# Set strong local keys and your absolute repo path in .env before starting
make bootstrap
sudo docker compose up -d
```

### 2. Local Service Endpoints

* **API Gateway (Nginx on port 80):**
  * Project 1 (Fraud) Swagger UI: `http://localhost/api/v1/fraud/docs`
  * Project 2 (Loan) Swagger UI: `http://localhost/api/v1/loan/docs`
* **Synthetic Simulation Listener (Nginx loopback `127.0.0.1:8081`):** Isolated endpoint (`/api/v1/loan-sim/predict`) used by `generate_traffic.py` and the Airflow DAG.
* **MLflow Tracking UI:** `http://127.0.0.1:5000`
* **Airflow 3.4 UI:** `http://127.0.0.1:8080`
* **Headless AML Streaming Worker (`mlops-aml-streaming`):** Runs a bounded 200,000-event pass (100k warmup + 100k out-of-sample scored when `AML_DAEMON_MODE=0`), writing reports, alerts, and DLQ records to `project-3-aml-streaming/artifacts/`.

### 3. Local Verification

```bash
conda activate mlops-lab
make lint
make security
make test
```

## 📜 License

This project is licensed under the MIT License — see the `LICENSE` file for details.
