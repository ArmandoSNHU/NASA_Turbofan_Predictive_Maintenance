# src/preprocess.py
import pandas as pd
from pathlib import Path

COLS = ["unit", "cycle",
        "op_setting_1", "op_setting_2", "op_setting_3",
        *[f"sensor_{i}" for i in range(1, 22)]]

def _resolve_path(candidate: str) -> Path:
    """Resolve to full dataset if present, else fall back to demo file."""
    p = Path(candidate)
    if p.exists():
        return p
    demo = Path("data/train_FD001_demo.txt")
    if demo.exists():
        print(f"⚠️ {p.name} not found. Using demo dataset instead -> {demo}")
        return demo
    raise FileNotFoundError(f"Missing dataset. Tried {p} and {demo}")

def load_fd001(filename: str = "data/train_FD001.txt") -> pd.DataFrame:
    """Load FD001 (whitespace-delimited)."""
    p = _resolve_path(filename)
    df = pd.read_csv(p, sep=r"\s+", header=None, names=COLS, engine="python")
    # Some sources add extra blank columns—trim to expected 26
    if df.shape[1] > len(COLS):
        df = df.iloc[:, :len(COLS)]
    return df

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life label based on max cycle per unit."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(lower=0)
    return df
