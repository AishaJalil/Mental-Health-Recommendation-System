"""
prepare_dass21.py
=================
Converts a downloaded Kaggle dataset into dass21_raw.csv so the app can use
real stress-profile data instead of purely synthetic scores.

SUPPORTED DATASETS
------------------
1. DASS-42 Raw Responses (most common Kaggle format)
   - Columns: Q1A, Q2A, … Q42A  (values 1-4, coded as 0-3 internally)
   - Script computes Depression / Anxiety / Stress subscale totals
   - File to download: search Kaggle for "DASS depression anxiety stress scale"
     Typical filename after extraction: data.csv

2. Student Stress Factors dataset
   - Columns include: anxiety_level, depression, stress_level (0-21 range)
   - Script scales these up to DASS-42 equivalent (0-42)
   - File to download: search Kaggle for "student stress factors"
     Typical filename: StressLevelDataset.csv

3. Generic pre-computed (already in range 0-42)
   - Columns named exactly:  depression  anxiety  stress
   - OR:                     dass_depression  dass_anxiety  dass_stress
   - Script passes them through with no conversion

USAGE
-----
  python backend/data/prepare_dass21.py <path_to_downloaded_csv>

The script writes:
  backend/data/dass21_raw.csv   ← columns: depression, anxiety, stress (0-42)

Then restart the Flask server — it will automatically use the real data.
"""

import sys
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# DASS-42 subscale item numbers (1-indexed question numbers)
# ---------------------------------------------------------------------------
DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
ANXIETY_ITEMS    = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
STRESS_ITEMS     = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]


def _compute_from_raw_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute subscale scores from individual Q1A-Q42A columns.
    Kaggle DASS raw files use values 1-4; the real scale is 0-3.
    """
    print("  Detected raw DASS-42 item columns (Q1A-Q42A).")

    # Some datasets use 1-4, others already use 0-3.
    # Detect by checking max across all Q columns.
    q_cols = [c for c in df.columns if c.startswith("Q") and c.endswith("A")]
    max_val = df[q_cols].max().max()
    offset = 1 if max_val > 3 else 0
    if offset:
        print("  Values appear to be 1-4 scale — subtracting 1 to normalise to 0-3.")

    rows = []
    for _, row in df.iterrows():
        def item_sum(item_list):
            total = 0
            for i in item_list:
                col = f"Q{i}A"
                if col in df.columns:
                    total += max(0, int(row[col]) - offset)
            return int(np.clip(total, 0, 42))

        rows.append({
            "depression": item_sum(DEPRESSION_ITEMS),
            "anxiety":    item_sum(ANXIETY_ITEMS),
            "stress":     item_sum(STRESS_ITEMS),
        })
    return pd.DataFrame(rows)


def _from_student_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Student Stress Factors dataset has 0-21 range columns.
    Scale up to 0-42 so severity cutoffs stay correct.
    """
    print("  Detected Student Stress Factors format (0-21 scale).")
    out = pd.DataFrame()
    out["depression"] = (df["depression"] * 2).clip(0, 42).astype(int)
    out["anxiety"]    = (df["anxiety_level"] * 2).clip(0, 42).astype(int)
    out["stress"]     = (df["stress_level"] * 2).clip(0, 42).astype(int)
    return out


def _from_precomputed(df: pd.DataFrame) -> pd.DataFrame:
    """Already has depression/anxiety/stress or dass_* columns."""
    print("  Detected pre-computed score columns.")
    if {"depression", "anxiety", "stress"}.issubset(df.columns):
        out = df[["depression", "anxiety", "stress"]].copy()
    else:
        out = df[["dass_depression", "dass_anxiety", "dass_stress"]].copy()
        out.columns = ["depression", "anxiety", "stress"]
    return out.astype(int).clip(0, 42)


def prepare(input_path: str) -> None:
    print(f"\nReading: {input_path}")
    df = pd.read_csv(input_path)
    df = df.dropna(how="all")
    print(f"  Rows loaded: {len(df)}  |  Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    cols = set(df.columns)

    # ── Detect format ──────────────────────────────────────────────────
    if "Q1A" in cols and "Q42A" in cols:
        out = _compute_from_raw_items(df)

    elif {"anxiety_level", "depression", "stress_level"}.issubset(cols):
        out = _from_student_stress(df)

    elif {"depression", "anxiety", "stress"}.issubset(cols):
        out = _from_precomputed(df)

    elif {"dass_depression", "dass_anxiety", "dass_stress"}.issubset(cols):
        out = _from_precomputed(df)

    else:
        print("\nERROR: Could not detect dataset format.")
        print("Expected one of:")
        print("  • Q1A … Q42A               (DASS-42 raw items)")
        print("  • depression, anxiety, stress")
        print("  • dass_depression, dass_anxiety, dass_stress")
        print("  • anxiety_level, depression, stress_level  (Student Stress Factors)")
        sys.exit(1)

    # ── Drop rows where everything is 0 (probably invalid/test rows) ──
    out = out[(out["depression"] + out["anxiety"] + out["stress"]) > 0]

    # ── Save ──────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dass21_raw.csv")
    out.to_csv(out_path, index=False)

    print(f"\nSaved {len(out)} records → {out_path}")
    print(f"  Depression  avg={out['depression'].mean():.1f}  max={out['depression'].max()}")
    print(f"  Anxiety     avg={out['anxiety'].mean():.1f}  max={out['anxiety'].max()}")
    print(f"  Stress      avg={out['stress'].mean():.1f}  max={out['stress'].max()}")
    print("\nDone! Now restart the Flask server (python backend/app.py).")
    print("The app will automatically load the real DASS data for the first 100 users.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    prepare(sys.argv[1])
