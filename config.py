from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "warehouse.db"
REPORTS = ROOT / "reports"
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
SEED = 42
