import argparse

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from utils import LABELS, ensure_dir, load_training_data, save_meta


def main():
    parser = argparse.ArgumentParser(description="Train toxicity classifier")
    parser.add_argument("--fast", action="store_true", help="Use smaller dataset")
    args = parser.parse_args()

    print("Loading data...")
    X, y = load_training_data(fast=args.fast)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=500, solver="liblinear")
                ),
            ),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)
    thresholds = {label: 0.5 for label in LABELS}
    preds = (probs >= 0.5).astype(int)

    per_label_f1 = {
        label: float(f1_score(y_val[label], preds[:, i], zero_division=0))
        for i, label in enumerate(LABELS)
    }
    macro_f1 = float(sum(per_label_f1.values()) / len(per_label_f1))

    ensure_dir("saved_model")
    model_path = "saved_model/model.joblib"
    meta_path = "saved_model/meta.json"

    joblib.dump(model, model_path)
    save_meta(
        meta_path,
        {
            "labels": LABELS,
            "thresholds": thresholds,
            "metrics": {
                "macro_f1": macro_f1,
                "per_label_f1": per_label_f1,
            },
        },
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Validation macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
