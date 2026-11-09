import subprocess, sys
for step in ["download", "etl", "train", "profit"]:
    print(f"\n=== {step} ==="); subprocess.run([sys.executable, f"src/{step}.py"], check=True)
print("\nDone. Launch dashboard:  streamlit run app.py")
