"""
dataset.py - downloads and prepares the UCI Bank Marketing dataset.
Auto-detects CSV separator. Falls back to realistic synthetic data.
"""
import os, io, zipfile, urllib.request, logging, yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

_SOURCES = [
    ("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv", False),
    ("https://archive.ics.uci.edu/static/public/222/bank+additional.zip", True),
]

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _sniff_sep(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return ";" if line.count(";") > line.count(",") else ","

def download_dataset(cfg):
    raw_dir  = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, cfg["dataset"]["filename"])
    if os.path.exists(filepath) and os.path.getsize(filepath) > 50_000:
        logger.info("Dataset present: %s", filepath)
        return filepath

    for url, is_zip in _SOURCES:
        tmp = filepath + ".tmp"
        try:
            logger.info("Trying: %s", url)
            urllib.request.urlretrieve(url, tmp)
            if is_zip:
                with zipfile.ZipFile(tmp, "r") as z:
                    csvs = sorted([n for n in z.namelist() if n.lower().endswith(".csv")],
                                  key=lambda n: -z.getinfo(n).file_size)
                    if not csvs:
                        os.remove(tmp); continue
                    with z.open(csvs[0]) as src, open(filepath, "wb") as dst:
                        dst.write(src.read())
                os.remove(tmp)
            else:
                os.replace(tmp, filepath)
            if os.path.getsize(filepath) > 50_000:
                sep = _sniff_sep(filepath)
                df  = pd.read_csv(filepath, sep=sep, nrows=3)
                if cfg["dataset"]["target_column"] in df.columns:
                    logger.info("Download OK: %s", filepath)
                    return filepath
            if os.path.exists(filepath): os.remove(filepath)
        except Exception as exc:
            logger.warning("Failed (%s): %s", url, exc)
            for f in (tmp, filepath):
                if os.path.exists(f): os.remove(f)

    logger.warning("All downloads failed. Generating synthetic dataset.")
    return _generate_synthetic(filepath)

def _generate_synthetic(filepath):
    rng = np.random.default_rng(42)
    n   = 41188
    duration = np.clip(rng.exponential(250, n).astype(int), 0, 3600)
    log_dur  = np.log1p(duration)
    p_yes    = np.clip(0.02 + 0.25*(log_dur/log_dur.max()), 0.02, 0.60)
    y        = np.where(rng.binomial(1, p_yes)==1, "yes", "no")
    jobs     = ["admin.","blue-collar","technician","services","management",
                "retired","self-employed","entrepreneur","housemaid","unemployed","student"]
    df = pd.DataFrame({
        "age": rng.integers(18,80,n), "job": rng.choice(jobs,n),
        "marital":   rng.choice(["married","single","divorced","unknown"],n),
        "education": rng.choice(["basic.4y","high.school","basic.9y","university.degree","unknown"],n),
        "default":   rng.choice(["no","unknown"],n),
        "housing":   rng.choice(["yes","no","unknown"],n),
        "loan":      rng.choice(["no","yes","unknown"],n),
        "contact":   rng.choice(["cellular","telephone"],n),
        "month":     rng.choice(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],n),
        "day_of_week": rng.choice(["mon","tue","wed","thu","fri"],n),
        "duration": duration, "campaign": rng.integers(1,15,n),
        "pdays":    rng.choice([999]+list(range(1,30)),n,p=[0.85]+[0.15/29]*29),
        "previous": rng.integers(0,5,n),
        "poutcome": rng.choice(["nonexistent","failure","success"],n),
        "emp.var.rate":   rng.choice([-1.8,-1.7,-1.1,1.1,1.4],n),
        "cons.price.idx": rng.uniform(92.2,94.8,n).round(3),
        "cons.conf.idx":  rng.uniform(-50.8,-26.9,n).round(1),
        "euribor3m":      rng.uniform(0.634,4.970,n).round(3),
        "nr.employed":    rng.choice([4963.6,5008.7,5076.2,5099.1,5191.0,5228.1],n),
        "y": y,
    })
    df.to_csv(filepath, sep=";", index=False)
    logger.info("Synthetic: rows=%d  pos_rate=%.3f", n, (y=="yes").mean())
    return filepath

def preprocess(filepath, cfg):
    sep        = _sniff_sep(filepath)
    target_col = cfg["dataset"]["target_column"]
    seed       = cfg["project"]["seed"]
    df = pd.read_csv(filepath, sep=sep)
    logger.info("Loaded: %d rows, %d cols  sep=%r", len(df), len(df.columns), sep)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found. Got: {list(df.columns)}")
    df[target_col] = (df[target_col]=="yes").astype(int)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].values
    encoders = {}
    for col in X.select_dtypes(include=["object","str"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    feature_names = X.columns.tolist()
    X = X.values.astype(np.float64)
    ts  = cfg["dataset"]["test_size"]
    vs  = cfg["dataset"]["val_size"]
    Xf, Xt, yf, yt = train_test_split(X, y, test_size=ts, random_state=seed, stratify=y)
    Xtr, Xv, ytr, yv = train_test_split(Xf, yf, test_size=vs/(1-ts), random_state=seed, stratify=yf)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv); Xt = sc.transform(Xt)
    logger.info("train=%d  val=%d  test=%d  pos=%.3f", len(Xtr), len(Xv), len(Xt), float(ytr.mean()))
    return {"X_train":Xtr,"y_train":ytr,"X_val":Xv,"y_val":yv,"X_test":Xt,"y_test":yt,
            "feature_names":feature_names,"encoders":encoders,"scaler":sc,
            "sample_ids":np.arange(len(Xtr))}

def save_processed(data, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    for k in ("X_train","y_train","X_val","y_val","X_test","y_test","sample_ids"):
        np.save(os.path.join(out_dir,f"{k}.npy"), data[k])
    pd.Series(data["feature_names"]).to_csv(os.path.join(out_dir,"feature_names.csv"),index=False,header=False)
    logger.info("Saved processed data to %s", out_dir)

def load_processed(out_dir="data/processed"):
    data = {k: np.load(os.path.join(out_dir,f"{k}.npy"))
            for k in ("X_train","y_train","X_val","y_val","X_test","y_test","sample_ids")}
    data["feature_names"] = pd.read_csv(os.path.join(out_dir,"feature_names.csv"),header=None).squeeze().tolist()
    return data