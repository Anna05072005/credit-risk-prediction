from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import joblib


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.columns.difference(num_cols)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = Pipeline(steps=[
        ("prep", preprocess),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])
    return model


def main() -> None:
    root = project_root()
    train_path = root / "data" / "processed" / "train.csv"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    y = df["late_pay"].astype(int)
    X = df.drop(columns=["late_pay"])

    model = build_pipeline(X)
    model.fit(X, y)

    out_path = model_dir / "logreg.joblib"
    joblib.dump(model, out_path)
    print("Saved model:", out_path)


if __name__ == "__main__":
    main()
