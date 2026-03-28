import argparse

import joblib
import numpy as np

from utils import LABELS, load_meta

MODEL_PATH = "saved_model/model.joblib"
META_PATH = "saved_model/meta.json"

DEMO_TEXTS = [
    "Thank you for helping with this issue.",
    "You are a total idiot and a loser.",
    "I will destroy you.",
    "This is neutral and polite language.",
]


def print_result(text, probs, thresholds):
    print("\nComment:")
    print(text)
    print("\nPredictions:")
    for i, label in enumerate(LABELS):
        p = float(probs[i])
        detected = p >= thresholds.get(label, 0.5)
        verdict = "DETECTED" if detected else "clean"
        print(f"- {label:14s} {p * 100:6.2f}%  {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--text", type=str, default=None, help="Single input text")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo texts")
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    meta = load_meta(META_PATH) or {}
    thresholds = meta.get("thresholds", {label: 0.5 for label in LABELS})

    if args.demo:
        texts = DEMO_TEXTS
    elif args.text:
        texts = [args.text]
    else:
        entered = input("Enter a comment: ").strip()
        texts = [entered]

    probs = model.predict_proba(texts)
    probs = np.asarray(probs)

    for i, text in enumerate(texts):
        print_result(text, probs[i], thresholds)


if __name__ == "__main__":
    main()
