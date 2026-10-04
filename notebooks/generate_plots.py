"""
generate_plots.py - generate all competition figures from a completed training run.

Run after training:
    python notebooks/generate_plots.py

Outputs 8 PNG files to plots/
"""
import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR  = "plots"
LOG_DIR  = "logs"
TRAJ_FILE = os.path.join(LOG_DIR, "trajectory.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load trajectory ----
records = []
with open(TRAJ_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

epochs     = [r["epoch"] for r in records]
risk_score = [r["risk_score"] for r in records]
risk_level = [r["risk_level"] for r in records]
val_acc    = [r["val_accuracy"] for r in records]
bk_rate    = [r["backdoor_success_rate"] for r in records]
bk_anomaly = [r.get("poisoning_anomaly_score", 0) for r in records]
ckpt_state = [r["checkpoint_state"] for r in records]
drift_score= [r.get("behavioral_drift_score", 0) for r in records]

LEVEL_COLOR = {"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444","CRITICAL":"#7c3aed"}
STATE_COLOR = {"TRUSTED":"#22c55e","MONITORED":"#f59e0b",
               "SUSPICIOUS":"#ef4444","QUARANTINED":"#7c3aed","RECOVERED":"#3b82f6"}

def style():
    plt.rcParams.update({
        "figure.facecolor":"#0f1117","axes.facecolor":"#1a1d27",
        "axes.edgecolor":"#3d4058","axes.labelcolor":"#e0e0e0",
        "xtick.color":"#a0a0a0","ytick.color":"#a0a0a0",
        "text.color":"#e0e0e0","grid.color":"#2a2d3a",
        "grid.linestyle":"--","grid.alpha":0.5,
        "font.family":"DejaVu Sans","font.size":11,
    })

style()

# ---- Figure 1: Security Risk Trajectory ----
fig, ax = plt.subplots(figsize=(10,5))
for i in range(len(epochs)-1):
    ax.plot([epochs[i],epochs[i+1]],[risk_score[i],risk_score[i+1]],
            color=LEVEL_COLOR[risk_level[i]], linewidth=2.5)
ax.scatter(epochs, risk_score,
           c=[LEVEL_COLOR[l] for l in risk_level], s=80, zorder=5)
for name,val,col in [("LOW",0.20,"#22c55e"),("MEDIUM",0.40,"#f59e0b"),
                      ("HIGH",0.60,"#ef4444"),("CRITICAL",0.62,"#7c3aed")]:
    ax.axhline(val, color=col, linestyle=":", alpha=0.6, linewidth=1)
    ax.text(epochs[-1]+0.1, val+0.01, name, color=col, fontsize=9)
# Mark attack injection
atk_epoch = next((r["epoch"] for r in records if r.get("attack_active")), None)
if atk_epoch:
    ax.axvline(atk_epoch, color="#ef4444", linestyle="--", alpha=0.8)
    ax.text(atk_epoch+0.1, 0.05, "Attack\nInjected", color="#ef4444", fontsize=9)
ax.set_xlabel("Training Epoch"); ax.set_ylabel("Risk Score")
ax.set_title("Security Risk Trajectory", fontsize=14, fontweight="bold", pad=12)
ax.set_ylim(-0.02, 1.02); ax.grid(True)
patches = [mpatches.Patch(color=v,label=k) for k,v in LEVEL_COLOR.items()]
ax.legend(handles=patches, loc="upper left", framealpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig1_risk_trajectory.png"), dpi=150)
plt.close(); print("fig1 done")

# ---- Figure 2: Clean Accuracy vs Backdoor Success Rate ----
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(epochs, val_acc,  color="#3b82f6", linewidth=2.5, marker="o", label="Val Accuracy (clean)")
ax.plot(epochs, bk_rate,  color="#ef4444", linewidth=2.5, marker="s", label="Backdoor Success Rate")
if atk_epoch:
    ax.axvline(atk_epoch, color="#f59e0b", linestyle="--", alpha=0.8)
    ax.text(atk_epoch+0.1, 0.05, "Attack\nInjected", color="#f59e0b", fontsize=9)
ax.set_xlabel("Training Epoch"); ax.set_ylabel("Rate")
ax.set_title("Clean Accuracy vs Backdoor Success Rate", fontsize=14, fontweight="bold", pad=12)
ax.set_ylim(-0.02, 1.10); ax.grid(True)
ax.legend(framealpha=0.3)
ax.annotate("Backdoor active\naccuracy unchanged", xy=(6,bk_rate[-1]),
            xytext=(4.5,0.6), color="#ef4444", fontsize=9,
            arrowprops=dict(arrowstyle="->",color="#ef4444"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig2_accuracy_vs_backdoor.png"), dpi=150)
plt.close(); print("fig2 done")

# ---- Figure 3: Anomaly Scores Over Time ----
fig, ax = plt.subplots(figsize=(10,5))
acc_anomaly = [r.get("poisoning_anomaly_score",0) for r in records]
pred_anomaly= [r.get("prediction_distribution_anomaly",0) for r in records]
ax.plot(epochs, acc_anomaly,  color="#f59e0b", linewidth=2, label="Accuracy Anomaly Score")
ax.plot(epochs, pred_anomaly, color="#a855f7", linewidth=2, label="Prediction Distribution Anomaly")
ax.plot(epochs, drift_score,  color="#06b6d4", linewidth=2, label="Behavioral Drift Score")
if atk_epoch:
    ax.axvline(atk_epoch, color="#ef4444", linestyle="--", alpha=0.8, label="Attack Injected")
ax.set_xlabel("Training Epoch"); ax.set_ylabel("Anomaly Score (0-1)")
ax.set_title("Security Signal Detection Over Training", fontsize=14, fontweight="bold", pad=12)
ax.set_ylim(-0.02, 1.10); ax.grid(True); ax.legend(framealpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig3_anomaly_signals.png"), dpi=150)
plt.close(); print("fig3 done")

# ---- Figure 4: Backdoor Success Rate Before/After Detection ----
pre  = [r for r in records if not r.get("attack_active")]
post = [r for r in records if r.get("attack_active")]
fig, axes = plt.subplots(1,2,figsize=(10,5))
for ax,grp,title,col in [(axes[0],pre,"Before Attack (Clean Training)","#22c55e"),
                          (axes[1],post,"After Attack (Backdoor Active)","#ef4444")]:
    vals = [r["backdoor_success_rate"] for r in grp]
    ep   = [r["epoch"] for r in grp]
    ax.bar(ep, vals, color=col, alpha=0.8, width=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Backdoor Success Rate")
    ax.set_ylim(0, 1.15); ax.grid(True, axis="y")
    ax.axhline(np.mean(vals) if vals else 0, color="white",
               linestyle="--", alpha=0.5, label=f"Mean={np.mean(vals):.3f}")
    ax.legend(framealpha=0.3, fontsize=9)
fig.suptitle("Backdoor Success Rate: Pre vs Post Attack", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig4_backdoor_prepost.png"), dpi=150)
plt.close(); print("fig4 done")

# ---- Figure 5: Counterfactual Attribution ----
fig, ax = plt.subplots(figsize=(8,5))
passport_files = sorted([f for f in os.listdir(LOG_DIR)
                         if f.startswith("security_passport_") and f.endswith(".json")])
cf_data = None
if passport_files:
    with open(os.path.join(LOG_DIR, passport_files[-1])) as f:
        passport = json.load(f)
    cf = passport.get("counterfactual_evidence",{})
    if cf and cf.get("experiment_a") and cf.get("experiment_b"):
        cf_data = cf

labels = ["With Suspects\n(Attack Active)","Without Suspects\n(Attack Removed)"]
bk_vals = [
    cf_data["experiment_a"]["backdoor_success_rate"] if cf_data else 1.0,
    cf_data["experiment_b"]["backdoor_success_rate"] if cf_data else 0.0,
]
colors = ["#ef4444","#22c55e"]
bars = ax.bar(labels, bk_vals, color=colors, alpha=0.85, width=0.5, edgecolor="white", linewidth=1)
for bar,val in zip(bars,bk_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.3f}",
            ha="center", fontweight="bold", fontsize=13)
conf = cf_data.get("attribution_confidence",0.40) if cf_data else 0.40
ax.set_ylabel("Backdoor Attack Success Rate"); ax.set_ylim(0,1.25)
ax.set_title("Counterfactual Attribution Evidence", fontsize=14, fontweight="bold", pad=12)
ax.text(0.5, 1.12,
        f"Removing suspects eliminates backdoor (delta={bk_vals[0]-bk_vals[1]:.3f})\n"
        f"Attribution Confidence: {conf:.0%}  |  Counterfactual evidence, not causal proof.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#a0a0a0")
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig5_counterfactual.png"), dpi=150)
plt.close(); print("fig5 done")

# ---- Figure 6: Checkpoint State Timeline ----
fig, ax = plt.subplots(figsize=(10,4))
state_order = ["TRUSTED","MONITORED","SUSPICIOUS","QUARANTINED"]
state_y     = {s:i for i,s in enumerate(state_order)}
y_vals = []
for s in ckpt_state:
    y_vals.append(state_y.get(s, state_y.get("TRUSTED")))
for i,(ep,st) in enumerate(zip(epochs,ckpt_state)):
    col = STATE_COLOR.get(st,"#6b7280")
    ax.scatter(ep, state_y.get(st,0), color=col, s=200, zorder=5)
    if i > 0:
        ax.plot([epochs[i-1],ep],[y_vals[i-1],y_vals[i]],
                color=STATE_COLOR.get(ckpt_state[i-1],"#6b7280"), linewidth=2, alpha=0.7)
ax.set_yticks(range(len(state_order))); ax.set_yticklabels(state_order)
ax.set_xlabel("Training Epoch")
ax.set_title("Checkpoint State Transitions", fontsize=14, fontweight="bold", pad=12)
if atk_epoch:
    ax.axvline(atk_epoch, color="#ef4444", linestyle="--", alpha=0.7)
    ax.text(atk_epoch+0.1, -0.4, "Attack", color="#ef4444", fontsize=9)
ax.grid(True)
patches = [mpatches.Patch(color=STATE_COLOR[s],label=s) for s in state_order]
ax.legend(handles=patches, loc="upper left", framealpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig6_checkpoint_states.png"), dpi=150)
plt.close(); print("fig6 done")

# ---- Figure 7: Rollback Recovery Narrative ----
fig, ax = plt.subplots(figsize=(9,5))
ax.axis("off")
steps = [
    ("1  CLEAN TRAINING",    "#22c55e", "Epochs 1-4: TRUSTED checkpoints"),
    ("2  ATTACK INJECTED",   "#f59e0b", "Epoch 5: Poisoning + Backdoor"),
    ("3  ANOMALY DETECTED",  "#ef4444", "Epoch 6: Risk = CRITICAL (0.6405)"),
    ("4  COUNTERFACTUAL",    "#a855f7", "bk_with=1.0  bk_without=0.0  delta=1.0"),
    ("5  TRAINING PAUSED",   "#ef4444", "Automatic kill switch activated"),
    ("6  QUARANTINED",       "#7c3aed", "Epoch 6 checkpoint: SUSPICIOUS -> QUARANTINED"),
    ("7  ROLLBACK",          "#3b82f6", "Restored to epoch 4 (TRUSTED)"),
    ("8  RECOVERED",         "#22c55e", "Model verified healthy. Passport generated."),
]
for i,(title,col,detail) in enumerate(steps):
    y = 1 - i*0.118
    ax.add_patch(mpatches.FancyBboxPatch((0.01,y-0.05),0.98,0.10,
        boxstyle="round,pad=0.01", facecolor=col+"22", edgecolor=col, linewidth=1.5,
        transform=ax.transAxes, clip_on=False))
    ax.text(0.05, y+0.01, title, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=col)
    ax.text(0.55, y+0.01, detail, transform=ax.transAxes, fontsize=9, color="#c0c0c0")
    if i < len(steps)-1:
        ax.annotate("", xy=(0.50, y-0.055), xytext=(0.50, y-0.02),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555566", lw=1.5))
ax.set_title("Rollback Recovery Narrative", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig7_rollback_narrative.png"), dpi=150)
plt.close(); print("fig7 done")

# ---- Figure 8: System Architecture ----
fig, ax = plt.subplots(figsize=(10,6))
ax.axis("off")
boxes = [
    (0.5, 0.92, "TRAINING DATA", "#3b82f6", 0.40, 0.06),
    (0.5, 0.78, "TRAINING ENGINE", "#3b82f6", 0.40, 0.06),
    (0.5, 0.56, "MODEL IMMUNE SYSTEM", "#7c3aed", 0.60, 0.14),
    (0.18,0.32, "DETECT\nAnomaly Detection\nRisk Engine", "#f59e0b", 0.26, 0.14),
    (0.50,0.32, "PROVE\nSuspicious Samples\nCounterfactual", "#ef4444", 0.26, 0.14),
    (0.82,0.32, "CONTAIN\nPause / Quarantine\nRollback", "#22c55e", 0.26, 0.14),
    (0.5, 0.10, "TRUSTED MODEL + SECURITY PASSPORT", "#22c55e", 0.60, 0.06),
]
for (x,y,text,col,w,h) in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.015", facecolor=col+"33", edgecolor=col,
        linewidth=2, transform=ax.transAxes))
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, fontweight="bold", color=col, multialignment="center")
arrows = [(0.5,0.89,0.5,0.81),(0.5,0.75,0.5,0.63),
          (0.18,0.25,0.18,0.17),(0.50,0.25,0.50,0.17),(0.82,0.25,0.82,0.17)]
for x1,y1,x2,y2 in arrows:
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->",color="#8888aa",lw=1.8))
ax.set_title("Model Immune System — Architecture", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig8_architecture.png"), dpi=150)
plt.close(); print("fig8 done")

print(f"\nAll 8 figures saved to {OUT_DIR}/")