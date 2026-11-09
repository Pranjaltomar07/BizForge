"""Expected-value analysis of a retention campaign using calibrated out-of-fold probabilities.
Assumptions are explicit parameters (they are NOT estimated from this dataset):
  save_rate  - share of contacted would-be churners the offer actually retains
  cost       - cost per customer contacted (offer + handling)
  horizon    - months of future revenue credited for a saved customer
Expected net = sum_{contacted} p_churn * save_rate * monthly_charges * horizon - cost * n_contacted."""
import sqlite3, numpy as np, pandas as pd
from config import *

def curve(d: pd.DataFrame, save_rate=.3, cost=25.0, horizon=12):
    d = d.sort_values("p_churn", ascending=False).reset_index(drop=True)
    gain = d.p_churn * save_rate * d.monthly_charges * horizon
    net = gain.cumsum() - cost * (np.arange(len(d)) + 1)
    return pd.DataFrame({"contacted": np.arange(1, len(d) + 1), "expected_net": net.values})

def best(d, **kw):
    c = curve(d, **kw); i = int(c.expected_net.idxmax())
    everyone = float(c.expected_net.iloc[-1]); rng = np.random.default_rng(SEED)
    return {"best_k": int(c.contacted[i]), "best_pct_contacted": round(100 * (i + 1) / len(c), 1),
            "expected_net": round(float(c.expected_net[i]), 0), "contact_everyone_net": round(everyone, 0)}

def sensitivity(d):
    rows = []
    for sr in (.1, .2, .3, .4):
        for cost in (10, 25, 50):
            rows.append({"save_rate": sr, "cost": cost, **best(d, save_rate=sr, cost=cost)})
    return pd.DataFrame(rows)

def main():
    d = pd.read_sql("select * from oof_predictions", sqlite3.connect(DB_PATH))
    s = sensitivity(d); s.to_csv(REPORTS / "campaign_sensitivity.csv", index=False); print(s.to_string(index=False))

if __name__ == "__main__":
    main()
