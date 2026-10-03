# Model Immune System

**Protect AI while it is learning.**

> IBM Z Datathon 2026 — *AI Secured: Innovation Without Exposure*

---

## What it does

Model Immune System is a **training-time AI security and resilience system**.
It continuously observes a machine learning model while it trains, detects
poisoning attacks and backdoors, traces suspicious behavior back to the
responsible training data, verifies that cause using counterfactual
re-training experiments, and automatically pauses, quarantines, and rolls
back unsafe training.

The core security loop is:

```
DETECT → PROVE → CONTAIN
```

Traditional ML security asks *"Is this model safe?"* after training.
Model Immune System asks *"Is this model staying safe while it learns?"*

---

## Architecture

```
model-immune-system/
├── configs/            Configuration files
├── data/               Dataset (raw / processed / poisoned)
├── training/           Dataset loader, model factory, checkpointing, orchestrator
├── attacks/            Poisoning, backdoor, and drift simulation
├── monitoring/         Trajectory logging, anomaly detection, drift detection
├── detection/          Risk engine (aggregates all security signals)
├── attribution/        Suspicious sample tracker, counterfactual verification
├── response/           Policy engine (kill switch), rollback manager
├── passport/           Security Passport generator
├── api/                FastAPI REST backend
├── dashboard/          Streamlit Security Operations Center
└── tests/              Unit tests + end-to-end integration test
```

---

## Quick start

### 1 — Prerequisites

- Python 3.11 or 3.12
- pip

### 2 — Install

```bash
cd model-immune-system
pip install -r requirements.txt
```

### 3 — Run the training pipeline

```bash
python training/train.py
```

The first run downloads the UCI Bank Marketing dataset automatically.
If the download fails the pipeline falls back to a synthetic dataset.

### 4 — View the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 5 — Start the API

```bash
uvicorn api.main:app --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Run tests

```bash
# Unit tests
pytest tests/test_attacks.py tests/test_security.py -v

# Integration test (runs the full pipeline; takes ~30 s)
pytest tests/test_integration.py -v -m integration
```

---

## Docker

```bash
# Build and run everything
docker compose up --build

# Training only
docker compose run training

# Dashboard at http://localhost:8501
# API at http://localhost:8000/docs
```

---

## Configuration

All parameters live in [`configs/config.yaml`](configs/config.yaml).
Nothing is hard-coded.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `training.epochs` | 10 | Number of training rounds |
| `attacks.poisoning.inject_at_epoch` | 4 | When to inject the poisoning attack |
| `attacks.poisoning.poison_fraction` | 0.08 | Fraction of training data to corrupt |
| `attacks.backdoor.trigger_feature` | `duration` | Which feature the trigger modifies |
| `security.thresholds.critical` | 0.85 | Risk score that triggers automatic pause |
| `response.mode` | `automatic` | `automatic` or `manual` |

---

## IBM LinuxONE deployment

The entire application runs inside Docker containers on a standard Linux
kernel, making it directly portable to IBM LinuxONE (s390x).

```bash
# On LinuxONE — build for s390x
docker buildx build --platform linux/s390x -t model-immune-system:linuxone .

# Or use docker compose with the same compose file
docker compose up --build
```

All security telemetry (checkpoints, trajectory logs, response events,
Security Passport) remains inside the container/volume — the sensitive
model training metadata never leaves the controlled infrastructure boundary.
This demonstrates a key LinuxONE security value proposition: enterprise AI
training security inside a trusted compute environment.

---

## Security Passport

At the end of every training run the system generates an
**AI Security Passport** — a complete audit trail containing:

- Model identity and version
- Dataset provenance
- Training and attack configuration
- Full checkpoint history with state transitions
- Security risk trajectory
- Suspicious samples and batch IDs
- Counterfactual evidence
- Response actions taken
- Rollback information
- Final model security status

The passport is written to `logs/security_passport_<id>.json` and
`logs/security_passport_<id>.txt`.

---

## Demo narrative

```
1.  Start clean training on UCI Bank Marketing dataset
2.  Establish trusted baseline — checkpoints marked TRUSTED
3.  At epoch 4: inject poisoning + backdoor attack
4.  Model behavior begins shifting — backdoor success rate rises
5.  Security signals elevate — risk moves from LOW → MEDIUM → HIGH → CRITICAL
6.  Counterfactual verification: removing suspects reduces backdoor rate
7.  Attribution confidence confirmed
8.  Automatic pause: current checkpoint QUARANTINED
9.  Roll back to last TRUSTED checkpoint
10. Verify recovered model health
11. Generate Security Passport
```

---

## Team

IBM Z Datathon 2026

---

*Protect AI while it is learning.*
