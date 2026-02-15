from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = project_root()
    data_path = root / "data" / "cfcs_2014.csv"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print("Loaded:", data_path)
    print("Shape:", df.shape)

    target = "LATEPMTP"  # 1=late, 2=not late

    features = [
        "GREGION", "GNHHSIZE", "GGMRSTAT", "G2AGE", "SEX",
        "LF_G01", "HINCQUIN", "PINCQUIN",
        "IN_01A", "IN_01B", "IN_01C", "IN_01D", "IN_01E", "IN_01F", "IN_01G", "IN_01H", "IN_01I", "IN_D01",
        "G_ASSETS",
        "AD_11A", "AD_11B", "AD_11C", "AD_11D", "AD_11E", "AD_11F", "AD_11G",
    ]

    existing = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    print("Target exists:", target in df.columns)
    print("Features found:", len(existing))
    if missing:
        print("Missing features:", missing)

    data = df[[target] + existing].copy()

    # Map target
    y = pd.to_numeric(data[target], errors="coerce").map({1: 1, 2: 0})
    X = data.drop(columns=[target])

    # Keep only valid target rows
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)

    print("Final modeling data:")
    print("X shape:", X.shape)
    print("y distribution:", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df["late_pay"] = y_train.values

    test_df = X_test.copy()
    test_df["late_pay"] = y_test.values

    train_out = out_dir / "train.csv"
    test_out = out_dir / "test.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("Saved:", train_out)
    print("Saved:", test_out)


if __name__ == "__main__":
    main()
