import sqlite3, json, sys
from pathlib import Path
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import DB_PATH, REPORTS
from profit import curve, best

st.set_page_config("BizForge - Telco churn", layout="wide")
con = sqlite3.connect(DB_PATH); m = json.loads((REPORTS / "metrics.json").read_text())
oof = pd.read_sql("select * from oof_predictions", con)
st.title("BizForge - Telecom customer churn")
st.caption("Real data: IBM Telco Customer Churn sample (7,043 customers). Model scored once on an untouched 20% test set.")
t1, t2, t3 = st.tabs(["Overview", "Model", "Retention campaign"])
with t1:
    c = st.columns(3); c[0].metric("Customers", f"{len(oof):,}"); c[1].metric("Churn rate", f"{m['churn_rate']:.1%}")
    c[2].metric("Monthly revenue", f"${oof.monthly_charges.sum():,.0f}")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(pd.read_sql("select * from mart_churn_by_contract", con), x="segment", y="churn_rate", title="Churn rate by contract type"), width="stretch")
    t = pd.read_sql("select * from mart_churn_by_tenure", con)
    b.plotly_chart(px.bar(t, x="segment", y="churn_rate", category_orders={"segment": ["0-5", "6-11", "12-23", "24-47", "48+"]}, title="Churn rate by tenure (months)"), width="stretch")
with t2:
    c = st.columns(4); c[0].metric("Test ROC-AUC", m["test_auc"], f"95% CI {m['test_auc_95ci'][0]}-{m['test_auc_95ci'][1]}", delta_color="off")
    c[1].metric("PR-AUC", m["test_pr_auc"], f"baseline {m['pr_auc_baseline']}", delta_color="off")
    c[2].metric("Brier score", m["brier"], f"baseline {m['brier_baseline']}", delta_color="off"); c[3].metric("Top-decile lift", f"{m['top_decile_lift']}x")
    st.subheader("Model comparison (5-fold CV on training data)"); st.dataframe(pd.read_csv(REPORTS / "cv_results.csv"), hide_index=True)
    a, b = st.columns(2)
    cal = pd.read_csv(REPORTS / "calibration.csv"); f = go.Figure([go.Scatter(x=cal.mean_pred, y=cal.actual, mode="lines+markers", name="model"),
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash"))]); f.update_layout(title="Calibration (test set deciles)", xaxis_title="predicted", yaxis_title="actual")
    a.plotly_chart(f, width="stretch")
    pi = pd.read_csv(REPORTS / "permutation_importance.csv").head(10)[::-1]
    b.plotly_chart(px.bar(pi, x="auc_drop", y="feature", orientation="h", title="Permutation importance (AUC drop, test set)"), width="stretch")
with t3:
    st.info("Save rate, cost and horizon are assumptions, not estimated from this data. Adjust them to your business.")
    a, b, c = st.columns(3); sr = a.slider("Save rate", .05, .6, .3); cost = b.slider("Cost per contact ($)", 5, 100, 25); hz = c.slider("Months of revenue saved", 3, 24, 12)
    cv = curve(oof, sr, cost, hz); r = best(oof, save_rate=sr, cost=cost, horizon=hz)
    k = st.columns(3); k[0].metric("Best: contact top", f"{r['best_pct_contacted']}%"); k[1].metric("Expected net", f"${r['expected_net']:,.0f}"); k[2].metric("Contact everyone", f"${r['contact_everyone_net']:,.0f}")
    st.plotly_chart(px.line(cv, x="contacted", y="expected_net", title="Expected net value vs. number of customers contacted (highest risk first)"), width="stretch")
