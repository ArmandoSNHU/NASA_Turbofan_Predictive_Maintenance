# tools/simulate_cmapss_fd001.py
# ------------------------------------------------------------------
# Generate synthetic-but-realistic C-MAPSS FD001-style data.
# Writes: data/train_FD001.txt, data/test_FD001.txt, data/RUL_FD001.txt
# ------------------------------------------------------------------
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN_UNITS = 80
N_TEST_UNITS = 20
TRAIN_MIN_CYC, TRAIN_MAX_CYC = 150, 300
TEST_MIN_CYC, TEST_MAX_CYC = 80, 180

def gen_unit_series(unit_id: int, cycles: int, degrade_rate=0.002):
    unit = np.full((cycles, 1), unit_id, dtype=float)
    cyc = np.arange(1, cycles + 1, dtype=float).reshape(-1, 1)

    op1 = np.random.normal(0.0, 0.003, cycles).reshape(-1, 1)
    op2 = np.random.normal(0.0, 0.003, cycles).reshape(-1, 1)
    op3 = (100.0 + np.random.normal(0.0, 0.05, cycles)).reshape(-1, 1)

    base = np.array([
        520, 640, 1580, 1400, 14.6, 21.6, 553, 2388, 9046, 1.3,
        47.4, 522, 2388, 8138, 8.4, 0.03, 393, 2388, 100, 39.0, 23.4
    ], dtype=float)

    trend = np.array([
        +0.02, +0.03, -0.05, -0.03, +0.005, 0.0, +0.01, 0.0, -0.1, +0.0005,
        +0.02, +0.01, 0.0, -0.08, +0.004, 0.0, +0.02, 0.0, 0.0, +0.015, +0.012
    ]) * degrade_rate

    sensors = []
    for i in range(21):
        noise = np.random.normal(0, 1.0, cycles)
        s = base[i] + trend[i]*cyc.flatten() + noise
        sensors.append(s.reshape(-1, 1))
    sensors = np.hstack(sensors)

    return np.hstack([unit, cyc, op1, op2, op3, sensors])

def write_whitespace_txt(path: Path, mat: np.ndarray):
    with path.open("w", encoding="utf-8") as f:
        for row in mat:
            f.write(" ".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in row))
            f.write("\n")

def main():
    train_rows = []
    for u in range(1, N_TRAIN_UNITS + 1):
        max_cyc = int(rng.integers(TRAIN_MIN_CYC, TRAIN_MAX_CYC + 1))
        m = gen_unit_series(u, max_cyc)
        train_rows.append(m)
    train_mat = np.vstack(train_rows)
    write_whitespace_txt(DATA_DIR / "train_FD001.txt", train_mat)

    test_rows = []
    rul_list = []
    for u in range(1, N_TEST_UNITS + 1):
        fail_cyc = int(rng.integers(TRAIN_MIN_CYC, TRAIN_MAX_CYC + 1))
        observed = int(rng.integers(TEST_MIN_CYC, min(TEST_MAX_CYC, fail_cyc - 1)))
        m = gen_unit_series(u, observed)
        test_rows.append(m)
        rul_list.append(fail_cyc - observed)

    test_mat = np.vstack(test_rows)
    write_whitespace_txt(DATA_DIR / "test_FD001.txt", test_mat)

    with (DATA_DIR / "RUL_FD001.txt").open("w") as f:
        for r in rul_list:
            f.write(f"{r}\n")

    print("✅ Wrote train_FD001.txt, test_FD001.txt, RUL_FD001.txt")

if __name__ == "__main__":
    main()
