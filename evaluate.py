import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

from utils import LABELS, load_meta, load_training_data, save_meta

MODEL_PATH = "saved_model/model.joblib"
META_PATH = "saved_model/meta.json"


def tune_thresholds(y_true, probs):
    thresholds = {}
    for i, label in enumerate(LABELS):
        best_t = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.2, 0.8, 13):
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[label], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
        thresholds[label] = best_t
    return thresholds


def main():
    model = joblib.load(MODEL_PATH)
    meta = load_meta(META_PATH) or {"labels": LABELS, "thresholds": {l: 0.5 for l in LABELS}}

    X, y = load_training_data(fast=True)
    probs = model.predict_proba(X)

    tuned = tune_thresholds(y, probs)
    preds = np.column_stack([(probs[:, i] >= tuned[label]).astype(int) for i, label in enumerate(LABELS)])

    macro_f1 = f1_score(y.values, preds, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y.values, preds, target_names=LABELS, zero_division=0))

    meta["thresholds"] = tuned
    metrics = meta.get("metrics", {})
    metrics["eval_macro_f1"] = float(macro_f1)
    meta["metrics"] = metrics
    save_meta(META_PATH, meta)

    print("\nTuned thresholds:")
    for label in LABELS:
        print(f"- {label}: {tuned[label]:.2f}")


if __name__ == "__main__":
    main()
