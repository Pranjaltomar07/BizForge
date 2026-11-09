import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import etl, train, profit

def fake_csv(path, n=400, seed=0):
    rng = np.random.default_rng(seed); ten = rng.integers(0, 72, n)
    df = pd.DataFrame({"customerID": [f"C{i}" for i in range(n)], "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n), "Partner": rng.choice(["Yes", "No"], n), "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": ten, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No"], n), "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": rng.choice(["Yes", "No"], n), "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n), "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": np.round(rng.uniform(20, 110, n), 2)})
    df["TotalCharges"] = (df.MonthlyCharges * df.tenure).round(2).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "          # the real dataset has blank strings here
    p = 1 / (1 + np.exp(-(1.2 - .06 * ten + (df.Contract == "Month-to-month") * 1.0 - 1.5)))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No"); df.to_csv(path, index=False); return df

@pytest.fixture
def db(tmp_path):
    csv, dbp = tmp_path / "t.csv", tmp_path / "t.db"; raw = fake_csv(csv)
    etl.run(csv=csv, db=dbp); return dbp, raw

def test_etl_keeps_all_rows_and_handles_blank_total_charges(db):
    dbp, raw = db; con = sqlite3.connect(dbp)
    assert con.execute("select count(*) from mart_features").fetchone()[0] == len(raw)
    assert con.execute("select count(*) from stg_customers where total_charges is null").fetchone()[0] == 0
    assert con.execute("select min(total_charges) from stg_customers where tenure_months=0").fetchone()[0] == 0

def test_features_exclude_id_label_and_gender(db):
    _, X, y = train.load(db[0])
    assert not {"customer_id", "churned", "gender"} & set(X.columns) and set(y.unique()) <= {0, 1}

def test_pipeline_runs_and_beats_majority_baseline(db, tmp_path):
    res = train.main(db=db[0], out=tmp_path / "rep")
    assert res["test_auc"] > 0.6 and 0 <= res["brier"] <= 0.5
    oof = pd.read_sql("select * from oof_predictions", sqlite3.connect(db[0]))
    assert oof.p_churn.between(0, 1).all() and len(oof) == 400

def test_profit_curve_logic():
    d = pd.DataFrame({"p_churn": [.9, .5, .1], "monthly_charges": [100, 100, 100]})
    c = profit.curve(d, save_rate=.5, cost=10, horizon=10)       # gains 450, 250, 50 ; cost 10 each
    assert list(c.expected_net.round(0)) == [440, 680, 720]
    assert profit.best(d, save_rate=.5, cost=60, horizon=10)["best_k"] == 2   # third customer not worth $60
