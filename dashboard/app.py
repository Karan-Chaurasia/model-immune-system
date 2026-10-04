"""
app.py - Model Immune System: AI Security Operations Center
"""
import json, os, sys, time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from monitoring.trajectory import TrajectoryLog
from training.checkpoint   import list_checkpoints
from attribution.suspicious_samples import SuspiciousSampleTracker
from response.policy_engine import PolicyEngine

CONFIG_PATH = os.environ.get("MIS_CONFIG","configs/config.yaml")

st.set_page_config(
    page_title="Model Immune System",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, .stApp { background:#0f1117; color:#e0e0e0; }
.block-container { padding-top:1rem; max-width:1400px; }
.metric-card {
    background:#1a1d27; border:1px solid #2a2d3a;
    border-radius:10px; padding:16px 20px; text-align:center;
}
.metric-card .label { font-size:11px; color:#8888aa; text-transform:uppercase; letter-spacing:1px; }
.metric-card .value { font-size:28px; font-weight:700; margin-top:4px; }
.badge {
    display:inline-block; padding:5px 16px; border-radius:20px;
    font-weight:700; font-size:13px; letter-spacing:0.5px;
}
.section-header {
    border-bottom:1px solid #2a2d3a; padding-bottom:6px;
    margin-bottom:14px; font-size:15px; font-weight:600; color:#8888cc;
}
</style>
""", unsafe_allow_html=True)

RISK_COLOR  = {"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444","CRITICAL":"#7c3aed"}
STATE_COLOR = {"TRUSTED":"#22c55e","MONITORED":"#f59e0b",
               "SUSPICIOUS":"#ef4444","QUARANTINED":"#7c3aed","RECOVERED":"#3b82f6"}

@st.cache_data(ttl=4)
def load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_traj(cfg):
    return pd.DataFrame(TrajectoryLog(log_dir=cfg["training"]["log_dir"]).as_dicts())

def load_checkpoints(cfg): return list_checkpoints(cfg)

def load_suspicious(cfg):
    return SuspiciousSampleTracker(log_dir=cfg["training"]["log_dir"]).all()

def load_response(cfg):
    return PolicyEngine(cfg, log_dir=cfg["training"]["log_dir"]).all_events()

def load_passport(cfg):
    d = cfg["training"]["log_dir"]
    if not os.path.isdir(d): return None
    files = sorted([f for f in os.listdir(d)
                    if f.startswith("security_passport_") and f.endswith(".json")])
    if not files: return None
    with open(os.path.join(d, files[-1])) as f:
        return json.load(f)

def load_plots():
    d = "plots"
    if not os.path.isdir(d): return {}
    return {f: os.path.join(d,f) for f in sorted(os.listdir(d)) if f.endswith(".png")}

# ----------------------------------------------------------------
cfg = load_cfg()
traj_df    = load_traj(cfg)
checkpoints = load_checkpoints(cfg)
suspicious  = load_suspicious(cfg)
response    = load_response(cfg)
passport    = load_passport(cfg)
plots       = load_plots()

# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------
st.markdown("""
<div style="text-align:center;padding:20px 0 10px">
<div style="font-size:36px;font-weight:800;color:#8888ff;letter-spacing:2px">
  MODEL IMMUNE SYSTEM
</div>
<div style="font-size:15px;color:#8888aa;margin-top:4px;font-style:italic">
  Protect AI while it is learning &nbsp;|&nbsp; IBM Z Datathon 2026
</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    auto = st.toggle("Auto-refresh (4s)", value=False)
    st.divider()
    st.markdown("**Attack Configuration**")
    st.markdown(f"- Poison epoch: `{cfg['attacks']['poisoning']['inject_at_epoch']}`")
    st.markdown(f"- Poison fraction: `{cfg['attacks']['poisoning']['poison_fraction']:.0%}`")
    st.markdown(f"- Backdoor trigger: `{cfg['attacks']['backdoor']['trigger_feature']} = {cfg['attacks']['backdoor']['trigger_value']}`")
    st.markdown(f"- Critical threshold: `{cfg['security']['thresholds']['critical']}`")
    st.divider()
    st.markdown("**Response Mode**")
    st.markdown(f"`{cfg['response']['mode'].upper()}`")

# ----------------------------------------------------------------
# PANEL A: STATUS CARDS
# ----------------------------------------------------------------
st.markdown('<div class="section-header">A &nbsp; Model Status</div>', unsafe_allow_html=True)

if not traj_df.empty:
    latest = traj_df.iloc[-1]
    level  = latest["risk_level"]
    score  = latest["risk_score"]
    state  = latest["checkpoint_state"]
    fstatus = passport["final_model_status"] if passport else "UNKNOWN"

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        col = RISK_COLOR.get(level,"#888")
        st.markdown(f'<div class="metric-card"><div class="label">Risk Level</div>'
                    f'<div class="value" style="color:{col}">{level}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Risk Score</div>'
                    f'<div class="value">{score:.4f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">Val Accuracy</div>'
                    f'<div class="value">{latest["val_accuracy"]:.4f}</div></div>',
                    unsafe_allow_html=True)
    with c4:
        col2 = STATE_COLOR.get(state,"#888")
        st.markdown(f'<div class="metric-card"><div class="label">Checkpoint State</div>'
                    f'<div class="value" style="color:{col2}">{state}</div></div>',
                    unsafe_allow_html=True)
    with c5:
        fc = STATE_COLOR.get(fstatus,"#888")
        st.markdown(f'<div class="metric-card"><div class="label">Final Status</div>'
                    f'<div class="value" style="color:{fc}">{fstatus}</div></div>',
                    unsafe_allow_html=True)

    # Incident banner
    if fstatus == "RECOVERED":
        st.success("LAST INCIDENT: CRITICAL — Backdoor + Poisoning  |  RESPONSE: PAUSED → QUARANTINED → ROLLED BACK TO TRUSTED EPOCH")
    elif fstatus == "HEALTHY":
        st.success("No security incidents detected during this training run.")
else:
    st.info("No training data yet. Run `python training/train.py`")

st.divider()

# ----------------------------------------------------------------
# PANEL B+C: Charts
# ----------------------------------------------------------------
col_b, col_c = st.columns(2)

with col_b:
    st.markdown('<div class="section-header">B &nbsp; Security Risk Trajectory</div>', unsafe_allow_html=True)
    if not traj_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["risk_score"],
            mode="lines+markers", name="Risk Score",
            line=dict(color="#7c3aed",width=2),
            marker=dict(color=[RISK_COLOR.get(l,"#888") for l in traj_df["risk_level"]],size=10),
        ))
        for name,val,col in [("CRITICAL",0.62,"#7c3aed"),("HIGH",0.60,"#ef4444"),
                              ("MEDIUM",0.40,"#f59e0b"),("LOW",0.20,"#22c55e")]:
            fig.add_hline(y=val,line_dash="dot",line_color=col,
                          annotation_text=name,annotation_position="right",opacity=0.5)
        atk = traj_df[traj_df["attack_active"].notna()]
        if not atk.empty:
            fig.add_vline(x=atk.iloc[0]["epoch"],line_dash="dash",
                          line_color="#ef4444",annotation_text="Attack")
        fig.update_layout(height=300,margin=dict(t=10,b=30),
                          paper_bgcolor="#1a1d27",plot_bgcolor="#1a1d27",
                          font_color="#e0e0e0",xaxis_title="Epoch",yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

with col_c:
    st.markdown('<div class="section-header">C &nbsp; Clean Accuracy vs Backdoor Rate</div>', unsafe_allow_html=True)
    if not traj_df.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=traj_df["epoch"],y=traj_df["val_accuracy"],
            mode="lines+markers",name="Val Accuracy",line=dict(color="#3b82f6",width=2)))
        fig2.add_trace(go.Scatter(x=traj_df["epoch"],y=traj_df["backdoor_success_rate"],
            mode="lines+markers",name="Backdoor Rate",line=dict(color="#ef4444",width=2)))
        fig2.update_layout(height=300,margin=dict(t=10,b=30),
                           paper_bgcolor="#1a1d27",plot_bgcolor="#1a1d27",
                           font_color="#e0e0e0",xaxis_title="Epoch",yaxis_title="Rate")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ----------------------------------------------------------------
# PANEL D: Checkpoint Timeline
# ----------------------------------------------------------------
st.markdown('<div class="section-header">D &nbsp; Checkpoint Timeline</div>', unsafe_allow_html=True)
if checkpoints:
    rows = [{"Epoch":c["epoch"],"State":c["state"],
             "Val Acc":f'{c["metrics"].get("accuracy",0):.4f}',
             "Timestamp":c["timestamp"]} for c in checkpoints]
    df_c = pd.DataFrame(rows)
    st.dataframe(df_c.style.applymap(
        lambda v: f"color:{STATE_COLOR.get(v,'#888')};font-weight:bold"
        if v in STATE_COLOR else "", subset=["State"]),
        use_container_width=True, height=200)

st.divider()

# ----------------------------------------------------------------
# PANEL E+F: Suspicious Samples + Counterfactual
# ----------------------------------------------------------------
col_e, col_f = st.columns(2)

with col_e:
    st.markdown('<div class="section-header">E &nbsp; Suspicious Data</div>', unsafe_allow_html=True)
    if suspicious:
        rows = [{"Epoch":e["epoch"],"Type":e["attack_type"],
                 "N Samples":e["n_samples"],
                 "Sample IDs (first 5)":str(e["sample_ids"][:5])} for e in suspicious]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No suspicious samples recorded.")

with col_f:
    st.markdown('<div class="section-header">F &nbsp; Counterfactual Evidence</div>', unsafe_allow_html=True)
    cf = (passport or {}).get("counterfactual_evidence")
    if cf and cf.get("experiment_a") and cf.get("experiment_b"):
        ea,eb = cf["experiment_a"], cf["experiment_b"]
        m1,m2,m3 = st.columns(3)
        m1.metric("Attribution Confidence", f'{cf.get("attribution_confidence",0):.0%}')
        m2.metric("Backdoor WITH suspects",    f'{ea["backdoor_success_rate"]:.3f}')
        m3.metric("Backdoor WITHOUT suspects", f'{eb["backdoor_success_rate"]:.3f}',
                  delta=f'{eb["backdoor_success_rate"]-ea["backdoor_success_rate"]:+.3f}')
        fig_cf = go.Figure(data=[
            go.Bar(name="With Suspects",    x=["Accuracy","F1","Backdoor Rate"],
                   y=[ea["accuracy"],ea["f1"],ea["backdoor_success_rate"]],marker_color="#ef4444"),
            go.Bar(name="Without Suspects", x=["Accuracy","F1","Backdoor Rate"],
                   y=[eb["accuracy"],eb["f1"],eb["backdoor_success_rate"]],marker_color="#22c55e"),
        ])
        fig_cf.update_layout(barmode="group",height=220,margin=dict(t=10,b=20),
                             paper_bgcolor="#1a1d27",plot_bgcolor="#1a1d27",font_color="#e0e0e0")
        st.plotly_chart(fig_cf, use_container_width=True)
        st.caption(cf.get("note",""))
    else:
        st.info("No counterfactual results yet.")

st.divider()

# ----------------------------------------------------------------
# PANEL G: Response Log
# ----------------------------------------------------------------
st.markdown('<div class="section-header">G &nbsp; Response Log</div>', unsafe_allow_html=True)
if response:
    rows = [{"Epoch":e["epoch"],"Risk Level":e["risk_level"],
             "Score":f'{e["risk_score"]:.4f}',"Action":e["action"],
             "Timestamp":e["timestamp"]} for e in response]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)

st.divider()

# ----------------------------------------------------------------
# PANEL H: Evidence Graphs (from plots/ folder)
# ----------------------------------------------------------------
st.markdown('<div class="section-header">H &nbsp; Experiment Evidence</div>', unsafe_allow_html=True)
if plots:
    keys = sorted(plots.keys())
    rows_imgs = [keys[i:i+2] for i in range(0,len(keys),2)]
    for row in rows_imgs:
        cols = st.columns(len(row))
        for col,fname in zip(cols,row):
            with col:
                st.image(plots[fname], caption=fname.replace("_"," ").replace(".png","").title(),
                         use_column_width=True)
else:
    st.info("Run `python notebooks/generate_plots.py` to generate evidence graphs.")

st.divider()

# ----------------------------------------------------------------
# PANEL I: Security Passport
# ----------------------------------------------------------------
st.markdown('<div class="section-header">I &nbsp; Security Passport</div>', unsafe_allow_html=True)
if passport:
    c1,c2 = st.columns([3,1])
    with c1:
        st.json(passport, expanded=False)
    with c2:
        st.download_button("Download Passport (JSON)",
            data=json.dumps(passport,indent=2),
            file_name="security_passport.json", mime="application/json",
            use_container_width=True)
        st.markdown("**Summary**")
        for k,v in [("Model",passport["model"]["model_id"]),
                    ("Dataset",passport["dataset"]["dataset_id"]),
                    ("Status",passport["final_model_status"]),
                    ("Checkpoints",len(passport["checkpoint_history"])),
                    ("Suspicious Batches",len(passport["suspicious_samples"])),
                    ("Response Actions",len(passport["response_actions"]))]:
            st.markdown(f"- **{k}:** `{v}`")
else:
    st.info("Security Passport will appear here after training completes.")

if auto:
    time.sleep(4)
    st.cache_data.clear()
    st.rerun()