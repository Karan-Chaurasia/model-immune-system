"""
app.py — Streamlit Security Operations Center dashboard.

Panels:
  A. Model Status
  B. Security Risk Trajectory
  C. Training Performance
  D. Attack Monitoring
  E. Suspicious Data
  F. Counterfactual Evidence
  G. Checkpoint Timeline
  H. Response Log
  I. Security Passport
"""

import json
import os
import sys
import time
import yaml
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.trajectory import TrajectoryLog
from training.checkpoint import list_checkpoints
from attribution.suspicious_samples import SuspiciousSampleTracker
from response.policy_engine import PolicyEngine

CONFIG_PATH = os.environ.get("MIS_CONFIG", "configs/config.yaml")


@st.cache_data(ttl=3)
def load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_trajectory(cfg):
    log = TrajectoryLog(log_dir=cfg["training"]["log_dir"])
    return pd.DataFrame(log.as_dicts())


def load_checkpoints(cfg):
    return list_checkpoints(cfg)


def load_suspicious(cfg):
    t = SuspiciousSampleTracker(log_dir=cfg["training"]["log_dir"])
    return t.all()


def load_response_log(cfg):
    p = PolicyEngine(cfg, log_dir=cfg["training"]["log_dir"])
    return p.all_events()


def load_passport(cfg):
    log_dir = cfg["training"]["log_dir"]
    if not os.path.isdir(log_dir):
        return None
    files = sorted([
        f for f in os.listdir(log_dir)
        if f.startswith("security_passport_") and f.endswith(".json")
    ])
    if not files:
        return None
    with open(os.path.join(log_dir, files[-1])) as f:
        return json.load(f)


def risk_color(level: str) -> str:
    return {
        "LOW": "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH": "#ef4444",
        "CRITICAL": "#7c3aed",
    }.get(level, "#6b7280")


def state_color(state: str) -> str:
    return {
        "TRUSTED": "#22c55e",
        "MONITORED": "#f59e0b",
        "SUSPICIOUS": "#ef4444",
        "QUARANTINED": "#7c3aed",
        "RECOVERED": "#3b82f6",
    }.get(state, "#6b7280")


# ------------------------------------------------------------------ #
# Page setup                                                           #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Model Immune System",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Model Immune System — Security Operations Center")
st.caption("Protect AI while it is learning.  |  IBM Z Datathon 2026")

cfg = load_cfg()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    auto_refresh = st.toggle("Auto-refresh (3 s)", value=False)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("**Config**")
    st.json({
        "epochs": cfg["training"]["epochs"],
        "poison_epoch": cfg["attacks"]["poisoning"]["inject_at_epoch"],
        "backdoor_epoch": cfg["attacks"]["backdoor"]["inject_at_epoch"],
        "response_mode": cfg["response"]["mode"],
    })

# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #

traj_df = load_trajectory(cfg)
checkpoints = load_checkpoints(cfg)
suspicious = load_suspicious(cfg)
response_log = load_response_log(cfg)
passport = load_passport(cfg)

# ------------------------------------------------------------------ #
# Panel A — Model Status                                               #
# ------------------------------------------------------------------ #

st.subheader("A · Model Status")
if traj_df.empty:
    st.info("No training data yet. Run `python training/train.py` to start.")
else:
    latest = traj_df.iloc[-1]
    final_level = latest["risk_level"]
    final_score = latest["risk_score"]
    ckpt_state = latest["checkpoint_state"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = risk_color(final_level)
        st.markdown(
            f'<span class="status-badge" style="background:{color};color:#fff">{final_level}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Current Risk Level")
    with col2:
        st.metric("Risk Score", f"{final_score:.4f}")
    with col3:
        st.metric("Validation Accuracy", f"{latest['val_accuracy']:.4f}")
    with col4:
        color2 = state_color(ckpt_state)
        st.markdown(
            f'<span class="status-badge" style="background:{color2};color:#fff">{ckpt_state}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Checkpoint State")

    if passport:
        fstatus = passport.get("final_model_status", "UNKNOWN")
        fcolor = state_color(fstatus)
        st.markdown(
            f'**Final Model Status:** <span class="status-badge" '
            f'style="background:{fcolor};color:#fff;font-size:0.9rem">{fstatus}</span>',
            unsafe_allow_html=True,
        )

st.divider()

# ------------------------------------------------------------------ #
# Panel B — Security Risk Trajectory                                   #
# ------------------------------------------------------------------ #

st.subheader("B · Security Risk Trajectory")
if not traj_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=traj_df["epoch"], y=traj_df["risk_score"],
        mode="lines+markers", name="Risk Score",
        line=dict(color="#7c3aed", width=2),
        marker=dict(
            color=[risk_color(lv) for lv in traj_df["risk_level"]],
            size=8, line=dict(width=1, color="#fff"),
        ),
    ))
    for level, key in [("CRITICAL", "critical"), ("HIGH", "high"),
                       ("MEDIUM", "medium"), ("LOW", "low")]:
        thresh = cfg["security"]["thresholds"][key]
        fig.add_hline(y=thresh, line_dash="dot",
                      annotation_text=level, annotation_position="right",
                      line_color=risk_color(level), opacity=0.6)
    if not traj_df[traj_df["attack_active"].notna()].empty:
        first_attack = traj_df[traj_df["attack_active"].notna()].iloc[0]
        fig.add_vline(x=first_attack["epoch"], line_dash="dash",
                      line_color="#ef4444", annotation_text="Attack injected")
    fig.update_layout(
        xaxis_title="Epoch", yaxis_title="Risk Score",
        height=300, margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Awaiting training data.")

st.divider()

# ------------------------------------------------------------------ #
# Panels C & D side-by-side                                           #
# ------------------------------------------------------------------ #

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("C · Training Performance")
    if not traj_df.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["val_accuracy"],
            mode="lines+markers", name="Val Accuracy", line=dict(color="#3b82f6"),
        ))
        fig2.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["val_f1"],
            mode="lines+markers", name="Val F1", line=dict(color="#10b981"),
        ))
        fig2.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["val_auc"],
            mode="lines+markers", name="Val AUC", line=dict(color="#f59e0b"),
        ))
        fig2.update_layout(
            xaxis_title="Epoch", yaxis_title="Score",
            height=280, margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Awaiting training data.")

with col_d:
    st.subheader("D · Attack Monitoring")
    if not traj_df.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["val_accuracy"],
            mode="lines+markers", name="Clean Accuracy", line=dict(color="#22c55e"),
        ))
        fig3.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["backdoor_success_rate"],
            mode="lines+markers", name="Backdoor Success Rate", line=dict(color="#ef4444"),
        ))
        fig3.update_layout(
            xaxis_title="Epoch", yaxis_title="Rate",
            height=280, margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Awaiting training data.")

st.divider()

# ------------------------------------------------------------------ #
# Panel E — Suspicious Data                                            #
# ------------------------------------------------------------------ #

st.subheader("E · Suspicious Data")
if suspicious:
    rows = []
    for entry in suspicious:
        rows.append({
            "Epoch": entry["epoch"],
            "Attack Type": entry["attack_type"],
            "# Samples": entry["n_samples"],
            "Sample IDs (first 5)": str(entry["sample_ids"][:5]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No suspicious samples recorded.")

st.divider()

# ------------------------------------------------------------------ #
# Panel F — Counterfactual Evidence                                    #
# ------------------------------------------------------------------ #

st.subheader("F · Counterfactual Evidence")
if passport and passport.get("counterfactual_evidence"):
    cf = passport["counterfactual_evidence"]
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric("Attribution Confidence", f"{cf.get('attribution_confidence', 0):.2%}")
    if cf.get("experiment_a") and cf.get("experiment_b"):
        ea = cf["experiment_a"]
        eb = cf["experiment_b"]
        fig_cf = go.Figure(data=[
            go.Bar(name="With Suspects", x=["Accuracy", "F1", "Backdoor Rate"],
                   y=[ea["accuracy"], ea["f1"], ea["backdoor_success_rate"]],
                   marker_color="#ef4444"),
            go.Bar(name="Without Suspects", x=["Accuracy", "F1", "Backdoor Rate"],
                   y=[eb["accuracy"], eb["f1"], eb["backdoor_success_rate"]],
                   marker_color="#22c55e"),
        ])
        fig_cf.update_layout(
            barmode="group", height=280, margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_cf, use_container_width=True)
    st.caption(cf.get("note", ""))
else:
    st.info("No counterfactual results yet.")

st.divider()

# ------------------------------------------------------------------ #
# Panel G — Checkpoint Timeline                                        #
# ------------------------------------------------------------------ #

st.subheader("G · Checkpoint Timeline")
if checkpoints:
    rows_g = []
    for ckpt in checkpoints:
        rows_g.append({
            "Epoch": ckpt["epoch"],
            "State": ckpt["state"],
            "Val Accuracy": ckpt["metrics"].get("accuracy", "—"),
            "Timestamp": ckpt["timestamp"],
        })
    df_g = pd.DataFrame(rows_g)
    st.dataframe(
        df_g.style.applymap(
            lambda v: f"color: {state_color(v)};font-weight:bold"
            if v in ("TRUSTED", "MONITORED", "SUSPICIOUS", "QUARANTINED") else "",
            subset=["State"],
        ),
        use_container_width=True,
    )
else:
    st.info("No checkpoints saved yet.")

st.divider()

# ------------------------------------------------------------------ #
# Panel H — Response Log                                               #
# ------------------------------------------------------------------ #

st.subheader("H · Response Log")
if response_log:
    rows_h = []
    for evt in response_log:
        rows_h.append({
            "Epoch": evt["epoch"],
            "Risk Level": evt["risk_level"],
            "Risk Score": f"{evt['risk_score']:.4f}",
            "Action": evt["action"],
            "Mode": evt["mode"],
            "Timestamp": evt["timestamp"],
        })
    st.dataframe(pd.DataFrame(rows_h), use_container_width=True)
else:
    st.info("No response events yet.")

st.divider()

# ------------------------------------------------------------------ #
# Panel I — Security Passport                                          #
# ------------------------------------------------------------------ #

st.subheader("I · Security Passport")
if passport:
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        st.json(passport, expanded=False)
    with col_i2:
        st.download_button(
            "⬇ Download Passport (JSON)",
            data=json.dumps(passport, indent=2),
            file_name="security_passport.json",
            mime="application/json",
        )
        st.markdown("**Summary**")
        st.write(f"- Model: `{passport['model']['model_id']}`")
        st.write(f"- Dataset: `{passport['dataset']['dataset_id']}`")
        st.write(f"- Final Status: `{passport['final_model_status']}`")
        st.write(f"- Checkpoints: {len(passport['checkpoint_history'])}")
        st.write(f"- Suspicious Batches: {len(passport['suspicious_samples'])}")
        st.write(f"- Response Actions: {len(passport['response_actions'])}")
else:
    st.info("Security Passport will appear here after training completes.")

# ------------------------------------------------------------------ #
# Auto-refresh                                                         #
# ------------------------------------------------------------------ #

if auto_refresh:
    time.sleep(3)
    st.cache_data.clear()
    st.rerun()
