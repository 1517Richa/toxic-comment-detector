import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TOXIC_KEYWORDS = {
    "toxic": ["idiot", "stupid", "trash", "worthless"],
    "severe_toxic": ["die", "rot", "filth"],
    "obscene": ["damn", "hell", "crap"],
    "threat": ["kill", "hurt", "destroy"],
    "insult": ["moron", "loser", "fool"],
    "identity_hate": ["racist", "bigot", "hate your kind"],
}

NEUTRAL_SENTENCES = [
    "Thank you for your help.",
    "I appreciate your feedback.",
    "Can we discuss this calmly?",
    "This is a normal and respectful comment.",
    "Please share your thoughts when free.",
]


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_synthetic_data(n_rows: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        is_toxic = rng.random() < 0.45
        if not is_toxic:
            text = NEUTRAL_SENTENCES[int(rng.integers(0, len(NEUTRAL_SENTENCES)))]
            labels = [0] * len(LABELS)
        else:
            chosen_labels = [int(rng.random() < 0.35) for _ in LABELS]
            if sum(chosen_labels) == 0:
                chosen_labels[int(rng.integers(0, len(LABELS)))] = 1
            parts = []
            for idx, flag in enumerate(chosen_labels):
                if flag:
                    kw = TOXIC_KEYWORDS[LABELS[idx]]
                    parts.append(kw[int(rng.integers(0, len(kw)))])
            text = "You are " + " and ".join(parts) + "."
            labels = chosen_labels

        row = {"comment_text": text}
        row.update({label: labels[i] for i, label in enumerate(LABELS)})
        rows.append(row)

    df = pd.DataFrame(rows)
    return df["comment_text"].astype(str), df[LABELS].astype(int)


def load_training_data(data_path: str = "data/train.csv", fast: bool = False):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        missing = [c for c in ["comment_text", *LABELS] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {data_path}: {missing}")
        if fast:
            df = df.sample(min(len(df), 5000), random_state=42)
        return df["comment_text"].astype(str), df[LABELS].astype(int)

    rows = 800 if fast else 2500
    return generate_synthetic_data(n_rows=rows)


def save_meta(meta_path: str, payload: dict):
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_meta(meta_path: str):
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
