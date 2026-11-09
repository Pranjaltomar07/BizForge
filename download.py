"""Download the IBM Telco Customer Churn sample dataset (7,043 customers, 21 columns)."""
import urllib.request
from config import RAW_CSV, DATA_URL

def main():
    RAW_CSV.parent.mkdir(exist_ok=True)
    if RAW_CSV.exists():
        print(f"already have {RAW_CSV.name}"); return
    urllib.request.urlretrieve(DATA_URL, RAW_CSV); print(f"downloaded {RAW_CSV.name} ({RAW_CSV.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
