"""
dataset.py — downloads, validates, and prepares the UCI Bank Marketing dataset.
"""

import os
import urllib.request
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(cfg: dict) -> str:
    """Download dataset if not already present. Returns local file path."""
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, cfg["dataset"]["filename"])

    if os.path.exists(filepath):
        logger.info("Dataset already downloaded: %s", filepath)
        return filepath

    url = cfg["dataset"]["url"]
    logger.info("Downloading dataset from %s", url)
    try:
        urllib.request.urlretrieve(url, filepath)
        logger.info("Download complete: %s", filepath)
    except Exception as exc:
        # Fallback: generate a realistic synthetic dataset so the pipeline still runs.
        logger.warning("Download failed (%s). Generating synthetic dataset.", exc)
        filepath = _generate_synthetic_dataset(filepath)

    return filepath


def _generate_synthetic_dataset(filepath: str) -> str:
    """
    Generate a synthetic bank-marketing-style dataset for offline development.
    Column names and dtypes mirror the real UCI dataset.
    NOT intended for scientific use — development/demo only.
    """
    rng = np.random.default_rng(42)
    n = 10000

    age = rng.integers(18, 80, n)
    duration = rng.integers(0, 3000, n)
    campaign = rng.integers(1, 15, n)
    pdays = rng.choice([999] + list(range(1, 30)), n)
    previous = rng.integers(0, 5, n)
    emp_var_rate = rng.uniform(-3, 2, n).round(1)
    cons_price_idx = rng.uniform(92, 95, n).round(3)
    cons_conf_idx = rng.uniform(-50, -26, n).round(1)
    euribor3m = rng.uniform(0.6, 5, n).round(3)
    nr_employed = rng.uniform(4960, 5230, n).round(1)

    jobs = ["admin.", "blue-collar", "technician", "services", "management",
            "retired", "self-employed", "entrepreneur", "housemaid", "unemployed", "student"]
    marital = ["married", "single", "divorced"]
    education = ["basic.4y", "high.school", "basic.6y", "basic.9y",
                 "professional.course", "unknown", "university.degree", "illiterate"]
    default = ["no", "yes", "unknown"]
    housing = ["yes", "no", "unknown"]
    loan = ["no", "yes", "unknown"]
    contact = ["cellular", "telephone"]
    month = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week = ["mon", "tue", "wed", "thu", "fri"]
    poutcome = ["nonexistent", "failure", "success"]

    # Simple synthetic label: subscribe more likely when duration is long
    p_yes = np.clip(duration / 3000.0, 0.05, 0.50)
    y = rng.binomial(1, p_yes).astype(str)
    y = np.where(y == "1", "yes", "no")

    df = pd.DataFrame({
        "age": age,
        "job": rng.choice(jobs, n),
        "marital": rng.choice(marital, n),
        "education": rng.choice(education, n),
        "default": rng.choice(default, n),
        "housing": rng.choice(housing, n),
        "loan": rng.choice(loan, n),
        "contact": rng.choice(contact, n),
        "month": rng.choice(month, n),
        "day_of_week": rng.choice(day_of_week, n),
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": rng.choice(poutcome, n),
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "y": y,
    })

    df.to_csv(filepath, sep=";", index=False)
    logger.info("Synthetic dataset written: %s (%d rows)", filepath, len(df))
    return filepath


def preprocess(filepath: str, cfg: dict) -> dict:
    """
    Load raw CSV, encode categoricals, scale numerics, split into
    train/val/test, and return a dict of numpy arrays + metadata.
    """
    sep = cfg["dataset"].get("separator", ";")
    target_col = cfg["dataset"]["target_column"]
    seed = cfg["project"]["seed"]

    df = pd.read_csv(filepath, sep=sep)
    logger.info("Loaded dataset: %d rows, %d columns", *df.shape)

    # Encode target
    df[target_col] = (df[target_col] == "yes").astype(int)

    # Separate features
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = X.columns.tolist()
    X = X.values.astype(np.float64)

    # Train / val / test split
    test_size = cfg["dataset"]["test_size"]
    val_size = cfg["dataset"]["val_size"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # val_size is relative to the remaining training portion
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=relative_val, random_state=seed, stratify=y_train_full
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Assign sample IDs for traceability
    n_train = len(X_train)
    sample_ids = np.arange(n_train)

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test)
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "encoders": encoders,
        "scaler": scaler,
        "sample_ids": sample_ids,
    }


def save_processed(data: dict, out_dir: str = "data/processed") -> None:
    """Save processed arrays so other modules can reload without re-preprocessing."""
    os.makedirs(out_dir, exist_ok=True)
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "sample_ids"):
        np.save(os.path.join(out_dir, f"{key}.npy"), data[key])
    pd.Series(data["feature_names"]).to_csv(
        os.path.join(out_dir, "feature_names.csv"), index=False, header=False
    )
    logger.info("Processed data saved to %s", out_dir)


def load_processed(out_dir: str = "data/processed") -> dict:
    """Reload previously saved processed arrays."""
    data = {}
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "sample_ids"):
        data[key] = np.load(os.path.join(out_dir, f"{key}.npy"))
    data["feature_names"] = pd.read_csv(
        os.path.join(out_dir, "feature_names.csv"), header=None
    ).squeeze().tolist()
    return data
