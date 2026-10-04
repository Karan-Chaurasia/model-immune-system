"""
app.py - Model Immune System: AI Security Operations Center
Professional dark-theme SOC dashboard for IBM Z Datathon 2026.
"""
import json, os, sys, time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from monitoring.trajectory import TrajectoryLog
from training.checkpoint   import list_checkpoints
from attribution.suspicious_samples import SuspiciousSampleTracker
from response.policy_engine import PolicyEngine

CONFIG_PATH = os.environ.get("MIS_CONFIG", "configs/config.yaml")

st.set_page_config(
    page_title="Model Immune System — Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Base */
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
.block-container { padding: 1.2rem 2rem 2rem; max-width: 1440px; }

/* Metric cards */
.mis-card {
    background: #12151f;
    border: 1px solid #252840;
    border-radius: 12px;
    padding: 18px 16px 14px;
    text-align: center;
    height: 100%;
}
.mis-card .mis-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6668aa;
    margin-bottom: 6px;
}
.mis-card .mis-value {
    font-size: 26px;
    font-weight: 800;
    line-height: 1.1;
}

/* Section headers */
.mis-section {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5557aa;
    border-bottom: 1px solid #1e2038;
    padding-bottom: 6px;
    margin: 22px 0 14px;
}

/* Incident banner */
.mis-incident {
    background: linear-gradient(90deg, #1a0a2e, #12151f);
    border: 1px solid #7c3aed;
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 8px 0 16px;
    font-size: 13px;
}
.mis-healthy {
    background: linear-gradient(90deg, #0a1a0e, #12151f);
    border: 1px solid #22c55e;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 8px 0 16px;
    font-size: 13px;
}

/* Table styling */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colour maps ────────────────────────────────────────────────
RISK_COL  = {"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444","CRITICAL":"#7c3aed"}
STATE_COL = {"TRUSTED":"#22c55e","MONITORED":"#f59e0b",
             "SUSPICIOUS":"#ef4444","QUARANTINED":"#7c3aed","RECOVERED":"#3b82f6"}
PLOT_BG   = "#0d0f1a"
AXIS_COL  = "#3d3f5a"

def plotly_layout(height=300, legend=True):
    return dict(
        height=height, margin=dict(t=20,b=30,l=10,r=10),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color="#c0c2d8", size=11),
        xaxis=dict(gridcolor=AXIS_COL, zerolinecolor=AXIS_COL, title_font_size=11),
        yaxis=dict(gridcolor=AXIS_COL, zerolinecolor=AXIS_COL, title_font_size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10) if legend else None,
    )

# ── Data loaders ──────────────────────────────────────────────
@st.cache_data(ttl=4)
def load_cfg():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_data(ttl=4)
def load_traj(log_dir):
    return pd.DataFrame(TrajectoryLog(log_dir=log_dir).as_dicts())

@st.cache_data(ttl=4)
def load_ckpts(cfg):
    return list_checkpoints(cfg)

@st.cache_data(ttl=4)
def load_sus(log_dir):
    return SuspiciousSampleTracker(log_dir=log_dir).all()

@st.cache_data(ttl=4)
def load_resp(cfg):
    return PolicyEngine(cfg, log_dir=cfg["training"]["log_dir"]).all_events()

@st.cache_data(ttl=4)
def load_passport(log_dir):
    if not os.path.isdir(log_dir): return None
    files = sorted([f for f in os.listdir(log_dir)
                    if f.startswith("security_passport_") and f.endswith(".json")])
    if not files: return None
    with open(os.path.join(log_dir, files[-1]), encoding="utf-8") as f:
        return json.load(f)

def load_plots():
    d = "plots"
    if not os.path.isdir(d): return {}
    return {f: os.path.join(d,f) for f in sorted(os.listdir(d)) if f.endswith(".png")}

cfg      = load_cfg()
log_dir  = cfg["training"]["log_dir"]
traj_df  = load_traj(log_dir)
ckpts    = load_ckpts(cfg)
sus      = load_sus(log_dir)
resp     = load_resp(cfg)
passport = load_passport(log_dir)
plots    = load_plots()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Controls")
    if st.button("⟳  Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear(); st.rerun()
    auto = st.toggle("Auto-refresh every 4 s", value=False)

    st.divider()
    st.markdown("#### Attack Config")
    pa = cfg["attacks"]["poisoning"]
    ba = cfg["attacks"]["backdoor"]
    st.markdown(f"**Poison epoch** `{pa['inject_at_epoch']}`")
    st.markdown(f"**Poison fraction** `{pa['poison_fraction']:.0%}`")
    st.markdown(f"**Backdoor trigger** `{ba['trigger_feature']} = {ba['trigger_value']}`")
    st.markdown(f"**Trigger rate** `{ba['trigger_rate']:.0%}`")

    st.divider()
    st.markdown("#### Security Thresholds")
    th = cfg["security"]["thresholds"]
    for name, val, col in [("CRITICAL",th["critical"],"#7c3aed"),
                            ("HIGH",th["high"],"#ef4444"),
                            ("MEDIUM",th["medium"],"#f59e0b"),
                            ("LOW",th["low"],"#22c55e")]:
        st.markdown(f'<span style="color:{col};font-weight:700">{name}</span>'
                    f' &nbsp; ≥ `{val}`', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"**Response mode** `{cfg['response']['mode'].upper()}`")
    st.markdown(f"**Epochs** `{cfg['training']['epochs']}`")
    st.markdown(f"**Seed** `{cfg['project']['seed']}`")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 4px">
  <div style="font-size:32px;font-weight:900;letter-spacing:3px;color:#7779ff">
    🛡️ &nbsp;MODEL IMMUNE SYSTEM
  </div>
  <div style="font-size:13px;color:#6668aa;margin-top:3px;letter-spacing:1px">
    AI SECURITY OPERATIONS CENTER &nbsp;·&nbsp; IBM Z DATATHON 2026 &nbsp;·&nbsp;
    <em>Protect AI while it is learning</em>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION A — LIVE STATUS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="mis-section">A &nbsp;·&nbsp; Live Model Status</div>',
            unsafe_allow_html=True)

if traj_df.empty:
    st.info("No training data yet.  Run:  `python training/train.py`")
else:
    latest  = traj_df.iloc[-1]
    level   = latest["risk_level"]
    score   = latest["risk_score"]
    state   = latest["checkpoint_state"]
    fstatus = (passport or {}).get("final_model_status", "UNKNOWN")
    epoch   = int(latest["epoch"])

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    cards = [
        (c1, "Final Status",     fstatus,                   STATE_COL.get(fstatus,"#888")),
        (c2, "Risk Level",       level,                     RISK_COL.get(level,"#888")),
        (c3, "Risk Score",       f"{score:.4f}",            "#c0c2d8"),
        (c4, "Checkpoint State", state,                     STATE_COL.get(state,"#888")),
        (c5, "Val Accuracy",     f"{latest['val_accuracy']:.4f}", "#3b82f6"),
        (c6, "Epoch",            str(epoch),                "#6668aa"),
    ]
    for col, label, value, color in cards:
        with col:
            st.markdown(
                f'<div class="mis-card">'
                f'<div class="mis-label">{label}</div>'
                f'<div class="mis-value" style="color:{color}">{value}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    if fstatus == "RECOVERED":
        st.markdown("""
        <div class="mis-incident">
          <b style="color:#7c3aed">⚠ LAST INCIDENT:</b>
          &nbsp; CRITICAL — Backdoor + Poisoning detected at epoch 6
          &nbsp;·&nbsp;
          <b style="color:#22c55e">RESPONSE:</b>
          &nbsp; TRAINING PAUSED → EPOCH 6 QUARANTINED → ROLLED BACK TO TRUSTED EPOCH 4
          &nbsp;·&nbsp;
          <b style="color:#3b82f6">STATUS: RECOVERED</b>
        </div>""", unsafe_allow_html=True)
    elif fstatus == "HEALTHY":
        st.markdown("""
        <div class="mis-healthy">
          <b style="color:#22c55e">✓ HEALTHY</b>
          &nbsp;·&nbsp; No security incidents detected during this training run.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECTION B+C — CHARTS ROW 1
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="mis-section">B &nbsp;·&nbsp; Security Risk Trajectory</div>',
            unsafe_allow_html=True)

if not traj_df.empty:
    col_b, col_c = st.columns([3,2])

    with col_b:
        fig = go.Figure()
        # Shaded risk zones
        for lo, hi, col, name in [(0,0.20,"#22c55e","LOW"),(0.20,0.40,"#f59e0b","MEDIUM"),
                                   (0.40,0.62,"#ef4444","HIGH"),(0.62,1.0,"#7c3aed","CRITICAL")]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.06,
                          layer="below", line_width=0, annotation_text=name,
                          annotation_position="right", annotation_font_size=9,
                          annotation_font_color=col)
        # Line coloured by risk level
        for i in range(len(traj_df)-1):
            fig.add_trace(go.Scatter(
                x=traj_df["epoch"].iloc[i:i+2],
                y=traj_df["risk_score"].iloc[i:i+2],
                mode="lines", showlegend=False,
                line=dict(color=RISK_COL.get(traj_df["risk_level"].iloc[i],"#888"),width=3)))
        fig.add_trace(go.Scatter(
            x=traj_df["epoch"], y=traj_df["risk_score"], mode="markers",
            marker=dict(color=[RISK_COL.get(l,"#888") for l in traj_df["risk_level"]],
                        size=10, line=dict(width=1.5,color="#0d0f1a")),
            name="Epoch", showlegend=False,
            hovertemplate="Epoch %{x}<br>Score: %{y:.4f}<extra></extra>"))
        atk = traj_df[traj_df["attack_active"].notna()]
        if not atk.empty:
            fig.add_vline(x=atk.iloc[0]["epoch"], line_dash="dash",
                          line_color="#ef4444", line_width=1.5,
                          annotation_text="⚠ Attack Injected",
                          annotation_font_color="#ef4444", annotation_font_size=10)
        fig.update_layout(**plotly_layout(340),
                          xaxis_title="Training Epoch", yaxis_title="Risk Score",
                          yaxis_range=[-0.02,1.05])
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        # Risk level distribution donut
        level_counts = traj_df["risk_level"].value_counts().reset_index()
        level_counts.columns = ["level","count"]
        colors = [RISK_COL.get(l,"#888") for l in level_counts["level"]]
        fig_d = go.Figure(go.Pie(
            labels=level_counts["level"], values=level_counts["count"],
            hole=0.62, marker_colors=colors,
            textfont_size=11, showlegend=True,
            hovertemplate="%{label}: %{value} epochs<extra></extra>"))
        fig_d.add_annotation(text=f"<b>{len(traj_df)}</b><br>epochs",
                             x=0.5,y=0.5,showarrow=False,
                             font=dict(size=14,color="#c0c2d8"))
        fig_d.update_layout(**plotly_layout(340),
                            title=dict(text="Epochs by Risk Level",
                                       font=dict(size=12,color="#8888bb"),x=0.5))
        st.plotly_chart(fig_d, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SECTION D+E — CHARTS ROW 2
# ══════════════════════════════════════════════════════════════
if not traj_df.empty:
    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown('<div class="mis-section">C &nbsp;·&nbsp; Clean Accuracy vs Backdoor Rate</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=traj_df["epoch"],y=traj_df["val_accuracy"],
            mode="lines+markers",name="Val Accuracy (Clean)",
            line=dict(color="#3b82f6",width=2),marker=dict(size=7)))
        fig2.add_trace(go.Scatter(x=traj_df["epoch"],y=traj_df["backdoor_success_rate"],
            mode="lines+markers",name="Backdoor Success Rate",
            line=dict(color="#ef4444",width=2),marker=dict(size=7)))
        atk = traj_df[traj_df["attack_active"].notna()]
        if not atk.empty:
            fig2.add_vline(x=atk.iloc[0]["epoch"],line_dash="dash",
                           line_color="#f59e0b",line_width=1.5,
                           annotation_text="Attack",annotation_font_color="#f59e0b",
                           annotation_font_size=9)
        fig2.update_layout(**plotly_layout(280),
                           xaxis_title="Epoch",yaxis_title="Rate",yaxis_range=[-0.02,1.10])
        st.plotly_chart(fig2, use_container_width=True)

    with col_e:
        st.markdown('<div class="mis-section">D &nbsp;·&nbsp; Security Signals</div>',
                    unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=traj_df["epoch"],
            y=traj_df.get("behavioral_drift_score", pd.Series([0]*len(traj_df))),
            mode="lines+markers",name="Drift Score",
            line=dict(color="#06b6d4",width=2),marker=dict(size=7)))
        fig3.add_trace(go.Scatter(x=traj_df["epoch"],
            y=traj_df.get("prediction_distribution_anomaly", pd.Series([0]*len(traj_df))),
            mode="lines+markers",name="Prediction Anomaly",
            line=dict(color="#a855f7",width=2),marker=dict(size=7)))
        fig3.add_trace(go.Scatter(x=traj_df["epoch"],
            y=traj_df.get("poisoning_anomaly_score", pd.Series([0]*len(traj_df))),
            mode="lines+markers",name="Accuracy Anomaly",
            line=dict(color="#f59e0b",width=2),marker=dict(size=7)))
        fig3.update_layout(**plotly_layout(280),
                           xaxis_title="Epoch",yaxis_title="Score (0-1)",yaxis_range=[-0.02,1.10])
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# SECTION E — CHECKPOINT TIMELINE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="mis-section">E &nbsp;·&nbsp; Checkpoint Timeline</div>',
            unsafe_allow_html=True)
if ckpts:
    rows = []
    for c in ckpts:
        rows.append({
            "Epoch": c["epoch"],
            "State": c["state"],
            "Val Acc": round(c["metrics"].get("accuracy",0),4),
            "Risk Level": c.get("extra",{}).get("risk_level","—"),
            "Risk Score": round(c.get("extra",{}).get("risk_score",0),4),
            "Attack": c.get("extra",{}).get("attack_active") or "—",
            "Timestamp": c["timestamp"][:19].replace("T"," "),
        })
    df_c = pd.DataFrame(rows)

    def color_state(val):
        c = STATE_COL.get(val,"")
        return f"color:{c};font-weight:700" if c else ""
    def color_risk(val):
        c = RISK_COL.get(val,"")
        return f"color:{c};font-weight:600" if c else ""

    st.dataframe(
        df_c.style.map(color_state, subset=["State"])
                  .map(color_risk,  subset=["Risk Level"]),
        use_container_width=True, height=min(60+len(rows)*38, 320))

# ══════════════════════════════════════════════════════════════
# SECTION F+G — SUSPICIOUS SAMPLES + COUNTERFACTUAL
# ══════════════════════════════════════════════════════════════
col_f, col_g = st.columns(2)

with col_f:
    st.markdown('<div class="mis-section">F &nbsp;·&nbsp; Suspicious Data</div>',
                unsafe_allow_html=True)
    if sus:
        total_sus = sum(e["n_samples"] for e in sus)
        st.markdown(f'<div style="font-size:28px;font-weight:800;color:#ef4444">'
                    f'{total_sus:,}</div>'
                    f'<div style="font-size:11px;color:#6668aa;margin-bottom:12px">'
                    f'TOTAL SUSPICIOUS SAMPLES IDENTIFIED</div>', unsafe_allow_html=True)
        rows = [{"Epoch":e["epoch"],"Type":e["attack_type"].upper(),
                 "Samples":e["n_samples"],
                 "Sample IDs (first 5)":str(e["sample_ids"][:5])} for e in sus]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No suspicious samples recorded.")

with col_g:
    st.markdown('<div class="mis-section">G &nbsp;·&nbsp; Counterfactual Evidence</div>',
                unsafe_allow_html=True)
    cf = (passport or {}).get("counterfactual_evidence")
    if cf and cf.get("experiment_a") and cf.get("experiment_b"):
        ea, eb = cf["experiment_a"], cf["experiment_b"]
        bk_with    = ea["backdoor_success_rate"]
        bk_without = eb["backdoor_success_rate"]
        delta      = bk_with - bk_without
        conf       = cf.get("attribution_confidence", 0)

        m1,m2,m3 = st.columns(3)
        m1.metric("WITH suspects",    f"{bk_with:.3f}",    delta=None)
        m2.metric("WITHOUT suspects", f"{bk_without:.3f}", delta=f"{bk_without-bk_with:+.3f}")
        m3.metric("Attribution Conf", f"{conf:.0%}")

        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=["Backdoor Rate","Accuracy","F1"],
            y=[ea["backdoor_success_rate"],ea["accuracy"],ea.get("f1",0)],
            name="With Suspects", marker_color="#ef4444",
            text=[f"{v:.3f}" for v in [ea["backdoor_success_rate"],ea["accuracy"],ea.get("f1",0)]],
            textposition="outside", textfont=dict(size=11)))
        fig_cf.add_trace(go.Bar(
            x=["Backdoor Rate","Accuracy","F1"],
            y=[eb["backdoor_success_rate"],eb["accuracy"],eb.get("f1",0)],
            name="Without Suspects", marker_color="#22c55e",
            text=[f"{v:.3f}" for v in [eb["backdoor_success_rate"],eb["accuracy"],eb.get("f1",0)]],
            textposition="outside", textfont=dict(size=11)))
        fig_cf.update_layout(**plotly_layout(240),barmode="group",
                             yaxis_range=[0,1.3])
        st.plotly_chart(fig_cf, use_container_width=True)
        st.caption("⚠ Counterfactual evidence supporting attribution — not absolute causal proof.")
    else:
        st.info("No counterfactual results yet.")

# ══════════════════════════════════════════════════════════════
# SECTION H — RESPONSE LOG
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="mis-section">H &nbsp;·&nbsp; Response Log</div>',
            unsafe_allow_html=True)
if resp:
    rows = [{"Epoch":e["epoch"],
             "Risk Level":e["risk_level"],
             "Score":round(e["risk_score"],4),
             "Action":e["action"].upper().replace("_"," "),
             "Mode":e["mode"].upper(),
             "Time":e["timestamp"][:19].replace("T"," ")} for e in resp]
    df_r = pd.DataFrame(rows)
    def color_action(val):
        if "PAUSE" in val or "QUARANTINE" in val: return "color:#7c3aed;font-weight:700"
        if "REVIEW" in val: return "color:#ef4444;font-weight:600"
        if "MONITOR" in val: return "color:#f59e0b"
        return "color:#22c55e"
    st.dataframe(
        df_r.style.map(color_risk, subset=["Risk Level"])
                  .map(color_action, subset=["Action"]),
        use_container_width=True, height=min(60+len(rows)*38, 280))

# ══════════════════════════════════════════════════════════════
# SECTION I — EVIDENCE PLOTS
# ══════════════════════════════════════════════════════════════
if plots:
    st.markdown('<div class="mis-section">I &nbsp;·&nbsp; Experiment Evidence</div>',
                unsafe_allow_html=True)
    keys = sorted(plots.keys())
    for i in range(0, len(keys), 2):
        row = keys[i:i+2]
        cols = st.columns(len(row))
        for col, fname in zip(cols, row):
            label = fname.replace("fig","Fig ").replace("_"," ").replace(".png","").strip()
            col.image(plots[fname], caption=label, use_column_width=True)
else:
    st.markdown('<div class="mis-section">I &nbsp;·&nbsp; Experiment Evidence</div>',
                unsafe_allow_html=True)
    st.info("Run `python notebooks/generate_plots.py` to generate 8 evidence figures.")

# ══════════════════════════════════════════════════════════════
# SECTION J — SECURITY PASSPORT
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="mis-section">J &nbsp;·&nbsp; Security Passport</div>',
            unsafe_allow_html=True)
if passport:
    col_j1, col_j2 = st.columns([3,1])
    with col_j2:
        fstatus = passport.get("final_model_status","UNKNOWN")
        fc      = STATE_COL.get(fstatus,"#888")
        st.markdown(f'<div class="mis-card">'
                    f'<div class="mis-label">Final Status</div>'
                    f'<div class="mis-value" style="color:{fc}">{fstatus}</div>'
                    f'</div><br>', unsafe_allow_html=True)
        st.download_button("⬇ Download Passport",
            data=json.dumps(passport,indent=2),
            file_name="security_passport.json",
            mime="application/json",
            use_container_width=True)
        st.markdown("---")
        for k,v in [("Model",    passport["model"]["model_id"]),
                    ("Dataset",  passport["dataset"]["dataset_id"]),
                    ("Checkpoints", len(passport["checkpoint_history"])),
                    ("Suspicious Batches", len(passport["suspicious_samples"])),
                    ("Response Actions",   len(passport["response_actions"]))]:
            st.markdown(f"<small style='color:#6668aa'>{k}</small><br>"
                        f"<b style='color:#c0c2d8'>{v}</b><br>",
                        unsafe_allow_html=True)
    with col_j1:
        st.json(passport, expanded=False)
else:
    st.info("Security Passport will appear here after training completes.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#3d3f5a;padding:4px 0">'
    'Model Immune System &nbsp;·&nbsp; IBM Z Datathon 2026 &nbsp;·&nbsp; '
    'Team clauseX &nbsp;·&nbsp; <em>Protect AI while it is learning</em>'
    '</div>', unsafe_allow_html=True)

if auto:
    time.sleep(4)
    st.cache_data.clear()
    st.rerun()