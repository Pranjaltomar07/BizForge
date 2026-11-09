"""Model comparison, calibration and evaluation on the real Telco churn data.
Protocol: 80/20 stratified split. Models compared by 5-fold CV on the TRAIN part only; the
winner is refit, isotonic-calibrated, and scored ONCE on the untouched test part.
Test-AUC uncertainty is a bootstrap CI. Out-of-fold predictions for all customers are
saved for the campaign analysis (profit.py)."""
import json, sqlite3, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from config import *

CAT = ["Contract", "InternetService", "PaymentMethod"]

def load(db=DB_PATH):
    df = pd.read_sql("select * from mart_features", sqlite3.connect(db))
    X = pd.get_dummies(df.drop(columns=["customer_id", "churned"]), columns=CAT, dtype=int)
    return df, X, df.churned

def models():
    return {
        "majority_class": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=2000)),
        "random_forest": RandomForestClassifier(400, min_samples_leaf=8, n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.04, subsample=.8,
                                 colsample_bytree=.8, min_child_weight=5, random_state=SEED, eval_metric="logloss"),
    }

def boot_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): out.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(out, q)), 4) for q in (2.5, 97.5)]

def main(db=DB_PATH, out=REPORTS):
    out.mkdir(exist_ok=True); df, X, y = load(db)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED); rows = []
    for name, m in models().items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc"); pr = cross_val_score(m, Xtr, ytr, cv=cv, scoring="average_precision")
        rows.append({"model": name, "cv_auc_mean": s.mean(), "cv_auc_std": s.std(), "cv_pr_auc_mean": pr.mean()})
    cvr = pd.DataFrame(rows).round(4); cvr.to_csv(out / "cv_results.csv", index=False); print(cvr.to_string(index=False))
    best = cvr[cvr.model != "majority_class"].sort_values("cv_auc_mean", ascending=False).iloc[0].model
    cal = CalibratedClassifierCV(models()[best], cv=5, method="isotonic").fit(Xtr, ytr)
    p = cal.predict_proba(Xte)[:, 1]; base = ytr.mean()
    top = np.argsort(-p)[: int(len(p) * .1)]
    res = {"selected_model": best, "n_train": len(Xtr), "n_test": len(Xte), "churn_rate": round(float(y.mean()), 4),
           "test_auc": round(roc_auc_score(yte, p), 4), "test_auc_95ci": boot_ci(yte, p),
           "test_pr_auc": round(average_precision_score(yte, p), 4), "pr_auc_baseline": round(float(yte.mean()), 4),
           "brier": round(brier_score_loss(yte, p), 4), "brier_baseline": round(brier_score_loss(yte, np.full(len(yte), base)), 4),
           "top_decile_lift": round(float(yte.values[top].mean() / yte.mean()), 2),
           "top_decile_capture": round(float(yte.values[top].sum() / yte.sum()), 4)}
    print(json.dumps(res, indent=2)); (out / "metrics.json").write_text(json.dumps(res, indent=2))
    # calibration table (deciles of predicted probability)
    q = pd.DataFrame({"p": p, "y": yte.values}); q["bin"] = pd.qcut(q.p, 10, duplicates="drop")
    q.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(4).to_csv(out / "calibration.csv", index=False)
    # permutation importance on the held-out test set (uncalibrated best model refit for speed)
    fit = models()[best].fit(Xtr, ytr)
    pi = permutation_importance(fit, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
    pd.DataFrame({"feature": X.columns, "auc_drop": pi.importances_mean, "std": pi.importances_std}).sort_values(
        "auc_drop", ascending=False).round(4).to_csv(out / "permutation_importance.csv", index=False)
    # out-of-fold calibrated probabilities for every customer (used by profit.py / dashboard)
    oof = cross_val_predict(CalibratedClassifierCV(models()[best], cv=3, method="isotonic"), X, y, cv=cv, method="predict_proba")[:, 1]
    d = df[["customer_id", "churned", "monthly_charges", "tenure_months", "Contract"]].copy(); d["p_churn"] = oof
    d.to_sql("oof_predictions", sqlite3.connect(db), if_exists="replace", index=False)
    return res

if __name__ == "__main__":
    main()
