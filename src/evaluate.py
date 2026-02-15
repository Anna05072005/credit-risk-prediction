from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    accuracy_score,
)

import matplotlib.pyplot as plt
import joblib


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = project_root()
    test_path = root / "data" / "processed" / "test.csv"
    model_path = root / "models" / "logreg.joblib"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_path)
    y_test = df["late_pay"].astype(int)
    X_test = df.drop(columns=["late_pay"])

    model = joblib.load(model_path)

    pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n================ RESULTS ================")
    print("Confusion matrix:\n", cm)
    print(f"\nAccuracy: {acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

    total = tn + fp + fn + tp
    late_total = tp + fn
    not_late_total = tn + fp

    print("\n=========== HUMAN SUMMARY ===========")
    print(f"Out of {total} people in the test set:")
    print(f"• {tp} HAD late payments and were correctly predicted late.")
    print(f"• {fn} HAD late payments but were predicted not late (missed risk).")
    print(f"• {fp} did NOT have late payments but were predicted late (false alarm).")
    print(f"• {tn} did NOT have late payments and were correctly predicted not late.")
    print(f"\nROC AUC: {auc:.3f}")

    # Save confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print("Saved:", cm_path)

    # Save ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    roc_path = results_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print("Saved:", roc_path)


if __name__ == "__main__":
    main()
