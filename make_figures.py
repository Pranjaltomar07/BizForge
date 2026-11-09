import sys, json, sqlite3, ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent
B, O, G, INK, MUT = "#2a78d6", "#eb6834", "#9a9a95", "#1a1a19", "#6b6a66"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
  "axes.edgecolor": "#bbb", "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.titleweight": "bold",
  "axes.titlelocation": "left", "axes.titlesize": 13, "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": True,
  "grid.color": "#e8e8e4", "grid.linewidth": .7, "axes.axisbelow": True})
def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); print(p)

# ---- BizForge
b = R; img = b/"docs"/"img"
cal = pd.read_csv(b/"reports/calibration.csv")
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.plot([0, .85], [0, .85], color=G, ls="--", lw=1.5, label="Perfect calibration")
ax.plot(cal.mean_pred, cal.actual, color=B, lw=2, marker="o", ms=8, mec="white", mew=1.5, label="Model (test deciles)")
ax.set_xlabel("Mean predicted churn probability"); ax.set_ylabel("Observed churn rate")
ax.set_title("Predicted probabilities match reality"); ax.legend(frameon=False, loc="upper left"); save(fig, img/"calibration.png")

pi = pd.read_csv(b/"reports/permutation_importance.csv").head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh(pi.feature.str.replace("_", " "), pi.auc_drop, xerr=pi["std"], color=B, height=.6, error_kw={"ecolor": MUT, "lw": 1})
ax.set_xlabel("Drop in held-out ROC-AUC when feature is shuffled"); ax.set_title("Contract type and tenure drive churn"); ax.grid(axis="y", visible=False)
save(fig, img/"importance.png")

con = sqlite3.connect(b/"data/warehouse.db"); d = pd.read_sql("select * from oof_predictions", con).sort_values("p_churn", ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 4.6)); n = np.arange(1, len(d)+1)
for sr, cost, col, ls in [(.3, 25, B, "-"), (.1, 25, O, "-")]:
    net = (d.p_churn*sr*d.monthly_charges*12).cumsum().values - cost*n
    ax.plot(n/len(d)*100, net/1000, color=col, lw=2.2, label=f"{int(sr*100)}% save rate, ${cost}/contact")
    i = net.argmax(); ax.plot(n[i]/len(d)*100, net[i]/1000, "o", color=col, ms=9, mec="white", mew=2)
ax.axhline(0, color=G, lw=1)
ax.set_xlabel("% of customers contacted (riskiest first)"); ax.set_ylabel("Expected net value ($ thousands)")
ax.set_title("Targeting beats contacting everyone"); ax.legend(frameon=False); ax.text(.99, .03, "Scenario: save rate is assumed, not measured", transform=ax.transAxes, ha="right", fontsize=9, color=MUT)
save(fig, img/"campaign_value.png")

raw = pd.read_csv(b/"data/Telco-Customer-Churn.csv"); raw["c"] = (raw.Churn == "Yes")
fig, ax = plt.subplots(figsize=(6, 4.2)); g = raw.groupby("Contract").c.mean().reindex(["Month-to-month", "One year", "Two year"])*100
bars = ax.bar(g.index, g.values, color=[O, B, B], width=.55)
for x, v in zip(range(3), g.values): ax.text(x, v+1, f"{v:.1f}%", ha="center", color=INK, fontsize=11)
ax.set_ylabel("Churn rate (%)"); ax.set_title("Month-to-month customers churn 15x more"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 50)
save(fig, img/"churn_by_contract.png")

