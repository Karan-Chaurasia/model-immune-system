"""
main.py — FastAPI backend for Model Immune System.

Exposes the training pipeline and security data via REST endpoints
so the Streamlit dashboard (and any other client) can query live state.
"""

import json
import os
import sys
import yaml
import subprocess
import threading
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from training.checkpoint import list_checkpoints, get_latest_trusted
from monitoring.trajectory import TrajectoryLog
from attribution.suspicious_samples import SuspiciousSampleTracker
from response.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Immune System API",
    description="Training-time AI security and resilience system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.environ.get("MIS_CONFIG", "configs/config.yaml")

_training_process: subprocess.Popen | None = None
_training_lock = threading.Lock()


def _load_cfg() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "Model Immune System API"}


# ------------------------------------------------------------------ #
# Training control                                                     #
# ------------------------------------------------------------------ #

@app.post("/training/start", tags=["Training"])
def start_training(background_tasks: BackgroundTasks):
    """Launch the training pipeline as a background process."""
    global _training_process
    with _training_lock:
        if _training_process and _training_process.poll() is None:
            return {"status": "already_running", "pid": _training_process.pid}

        cmd = [sys.executable, "training/train.py", "--config", CONFIG_PATH]
        _training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"status": "started", "pid": _training_process.pid}


@app.get("/training/status", tags=["Training"])
def training_status():
    global _training_process
    if _training_process is None:
        return {"status": "not_started"}
    code = _training_process.poll()
    if code is None:
        return {"status": "running", "pid": _training_process.pid}
    return {"status": "finished", "exit_code": code}


# ------------------------------------------------------------------ #
# Security data                                                        #
# ------------------------------------------------------------------ #

@app.get("/trajectory", tags=["Security"])
def get_trajectory():
    cfg = _load_cfg()
    log = TrajectoryLog(log_dir=cfg["training"]["log_dir"])
    return log.as_dicts()


@app.get("/checkpoints", tags=["Security"])
def get_checkpoints():
    cfg = _load_cfg()
    return list_checkpoints(cfg)


@app.get("/checkpoints/trusted", tags=["Security"])
def get_trusted_checkpoint():
    cfg = _load_cfg()
    trusted = get_latest_trusted(cfg)
    if trusted is None:
        raise HTTPException(status_code=404, detail="No trusted checkpoint found.")
    return trusted


@app.get("/suspicious-samples", tags=["Security"])
def get_suspicious_samples():
    cfg = _load_cfg()
    tracker = SuspiciousSampleTracker(log_dir=cfg["training"]["log_dir"])
    return tracker.all()


@app.get("/response-log", tags=["Security"])
def get_response_log():
    cfg = _load_cfg()
    policy = PolicyEngine(cfg, log_dir=cfg["training"]["log_dir"])
    return policy.all_events()


# ------------------------------------------------------------------ #
# Security Passport                                                    #
# ------------------------------------------------------------------ #

@app.get("/passport", tags=["Passport"])
def get_passport():
    cfg = _load_cfg()
    log_dir = cfg["training"]["log_dir"]
    passport_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("security_passport_") and f.endswith(".json")]
    )
    if not passport_files:
        raise HTTPException(status_code=404, detail="No passport found. Run training first.")
    path = os.path.join(log_dir, passport_files[-1])
    with open(path) as f:
        return json.load(f)


@app.get("/passport/download", tags=["Passport"])
def download_passport():
    cfg = _load_cfg()
    log_dir = cfg["training"]["log_dir"]
    passport_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("security_passport_") and f.endswith(".json")]
    )
    if not passport_files:
        raise HTTPException(status_code=404, detail="No passport found. Run training first.")
    path = os.path.join(log_dir, passport_files[-1])
    return FileResponse(path, media_type="application/json", filename=passport_files[-1])
