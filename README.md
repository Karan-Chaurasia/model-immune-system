# Model Immune System

**Protect AI while it is learning.**

> IBM Z Datathon 2026 — Theme: *AI Secured: Innovation Without Exposure*
> Team: clauseX

---

## One-line pitch

> Traditional ML security asks whether a trained model is safe.
> **Model Immune System asks whether the model is staying safe while it learns.**

---

## The demo in 10 steps

```
1.  Start training on enterprise-style tabular data
2.  Establish trusted baseline (epochs 1-4: TRUSTED checkpoints)
3.  Inject data poisoning + backdoor attack at epoch 5
4.  Security system detects behavioral shift
5.  Risk escalates: LOW -> MEDIUM -> CRITICAL
6.  Counterfactual verification:
      bk_with_suspects    = 1.000
      bk_without_suspects = 0.000   (backdoor disappears)
7.  Training automatically paused
8.  Unsafe checkpoint quarantined
9.  Rollback to last TRUSTED checkpoint (epoch 4)
10. Security Passport generated
    Final status: RECOVERED
```

---

## Security loop

```
DETECT -> PROVE -> CONTAIN
```

| Step | What happens |
|---|---|
| **DETECT** | Z-score anomaly, PSI drift, backdoor probe measure behavioral change |
| **PROVE** | Counterfactual: remove suspects, re-evaluate; delta = 1.0 confirms attribution |
| **CONTAIN** | Kill switch pauses training, quarantines checkpoint, rolls back to TRUSTED |

---

## Verified run output

```
Epoch 1  risk=LOW      (0.0000)  bk_rate=0.0000  val_acc=0.6026  state=TRUSTED
Epoch 2  risk=LOW      (0.0000)  bk_rate=0.0000  val_acc=0.6026  state=TRUSTED
Epoch 3  risk=LOW      (0.0000)  bk_rate=0.0000  val_acc=0.6026  state=TRUSTED
Epoch 4  risk=LOW      (0.0000)  bk_rate=0.0000  val_acc=0.6026  state=TRUSTED

>>> Injecting POISONING attack at epoch 5 <<<
>>> Injecting BACKDOOR  attack at epoch 5 <<<

Epoch 5  risk=MEDIUM   (0.4500)  bk_rate=1.0000  val_acc=0.8242  state=MONITORED
Epoch 6  risk=CRITICAL (0.6405)  bk_rate=1.0000  val_acc=0.8242  state=SUSPICIOUS

CRITICAL risk at epoch 6 -- pausing training.
Checkpoint epoch=6: SUSPICIOUS -> QUARANTINED
Counterfactual: bk_with=1.0000  bk_without=0.0000  delta=1.0000  confidence=0.40
Rollback complete -> epoch=4  val_accuracy=0.6026

=== Complete  status=RECOVERED  acc=0.6057  f1=0.2714 ===
```

---

## Repository structure

```
model-immune-system/
  configs/           config.yaml (all parameters)
  training/          dataset.py, model.py, checkpoint.py, train.py
  attacks/           poisoning.py, backdoor.py
  monitoring/        trajectory.py, anomaly.py, drift.py
  detection/         risk_engine.py
  attribution/       suspicious_samples.py, counterfactual.py
  response/          policy_engine.py, rollback.py
  passport/          security_passport.py
  api/               main.py (FastAPI)
  dashboard/         app.py (Streamlit SOC)
  notebooks/         generate_plots.py
  tests/             test_attacks.py, test_security.py, test_edge_cases.py
  plots/             8 evidence figures (generated)
  Dockerfile
  docker-compose.yml
  requirements.txt
```

---

## Quick start

```bash
git clone https://github.com/Karan-Chaurasia/model-immune-system.git
cd model-immune-system
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python training\train.py

# Linux / Mac / LinuxONE
.venv/bin/pip install -r requirements.txt
.venv/bin/python training/train.py
```

### Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### API

```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

### Generate evidence plots

```bash
python notebooks/generate_plots.py
# Saves 8 PNG figures to plots/
```

---

## Tests

```bash
# Unit tests
pytest tests/test_attacks.py tests/test_security.py -v

# Edge cases + no-false-positive test
pytest tests/test_edge_cases.py -v

# Full integration test
pytest tests/test_integration.py -v -m integration
```

---

## Docker

```bash
docker compose up --build
# Dashboard: http://localhost:8501
# API:       http://localhost:8000/docs
```

---

## IBM LinuxONE

The entire system runs inside Docker containers on a standard Linux kernel,
making it directly portable to IBM LinuxONE (s390x).

```bash
# Build for LinuxONE s390x
docker buildx build --platform linux/s390x -t model-immune-system:linuxone .
docker compose up --build
```

**Why LinuxONE matters here:**
All security telemetry (checkpoints, trajectory logs, counterfactual evidence,
Security Passport) remains inside the controlled infrastructure boundary.
Model training security monitoring runs within the trusted enterprise environment
without exposing sensitive model internals to external systems.

---

## Configuration

All parameters are in `configs/config.yaml`. Nothing is hard-coded.

| Key | Default | Description |
|---|---|---|
| `training.epochs` | 12 | Training rounds |
| `attacks.poisoning.inject_at_epoch` | 5 | When to inject poisoning |
| `attacks.poisoning.poison_fraction` | 0.12 | Fraction of data corrupted |
| `attacks.backdoor.trigger_feature` | `duration` | Trigger feature |
| `attacks.backdoor.trigger_value` | -99.0 | Impossible scaled value |
| `security.thresholds.critical` | 0.62 | Score that triggers automatic pause |
| `response.mode` | `automatic` | `automatic` or `manual` |

---

## Security Passport

Every training run generates a full audit trail:
- Model identity and version
- Dataset provenance
- Attack configuration
- Checkpoint history with state transitions (TRUSTED/MONITORED/SUSPICIOUS/QUARANTINED)
- Security risk trajectory (all epochs)
- Suspicious sample/batch IDs
- Counterfactual evidence (with vs without suspects)
- All response actions
- Rollback information
- Final model status

Saved to `logs/security_passport_<id>.json` and `.txt`.

---

## Attribution note

The counterfactual result provides **evidence supporting attribution**, not
absolute causal proof. A confidence of 0.40 means: removing the suspected
samples causes the suspicious behavior to disappear. This is consistent with
those samples being the source, but the system clearly labels this as
*counterfactual evidence*, not certainty.

---

## Team clauseX — IBM Z Datathon 2026