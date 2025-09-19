# run_demo.py
# --------------------------------------------------------
# One-click runner: trains, predicts, and inspects model.
# Uses your existing scripts and the current Python interpreter.
# --------------------------------------------------------
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # use the same interpreter as this script

SCRIPTS = [
    ("Training", ROOT / "src" / "train.py"),
    ("Prediction", ROOT / "src" / "predict.py"),
    ("Inspection", ROOT / "src" / "inspect_model.py"),
]

def run_step(name, script_path):
    print(f"\n=== {name} :: {script_path} ===")
    if not script_path.exists():
        raise SystemExit(f"❌ Missing script: {script_path}")
    proc = subprocess.run([PY, str(script_path)], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"❌ {name} failed with exit code {proc.returncode}")
    print(f"✅ {name} completed.")

def main():
    # Ensure output folders exist
    (ROOT / "reports" / "figs").mkdir(parents=True, exist_ok=True)

    for name, script in SCRIPTS:
        run_step(name, script)

    # Summarize expected artifacts
    model = ROOT / "reports" / "rf_fd001.joblib"
    preds = ROOT / "reports" / "predictions_fd001.csv"
    fig   = ROOT / "reports" / "figs" / "feature_importances.png"

    print("\n--- Artifacts ---")
    print(f"Model:        {'✅' if model.exists() else '❌'} {model}")
    print(f"Predictions:  {'✅' if preds.exists() else '❌'} {preds}")
    print(f"Importances:  {'✅' if fig.exists() else '❌'} {fig}")
    print("\nAll done. If you don't have the full dataset, the code will fall back to the demo file.")

if __name__ == '__main__':
    main()
