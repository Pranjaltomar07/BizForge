"""Load the raw CSV into SQLite, then run the SQL models in order (staging -> marts)."""
import sqlite3, pandas as pd
from config import *
MODELS = ["stg_customers", "mart_features", "mart_churn_segments"]

def run(csv=RAW_CSV, db=DB_PATH, sql_dir=ROOT / "sql"):
    con = sqlite3.connect(db)
    pd.read_csv(csv, dtype=str).assign(SeniorCitizen=lambda d: d.SeniorCitizen.astype(int),
        tenure=lambda d: d.tenure.astype(int)).to_sql("raw_customers", con, if_exists="replace", index=False)
    for m in MODELS:
        con.executescript((sql_dir / f"{m}.sql").read_text())
    n = con.execute("select count(*) from mart_features").fetchone()[0]; con.commit(); con.close()
    print(f"ETL done: {n} customers in mart_features"); return n

if __name__ == "__main__":
    run()
