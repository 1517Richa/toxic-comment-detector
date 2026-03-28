import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import LABELS, load_meta

MODEL_PATH = "saved_model/model.joblib"
META_PATH = "saved_model/meta.json"

st.set_page_config(page_title="Toxic Comment Detector", layout="wide")
st.title("Toxic Comment Detector")
st.caption("Run train.py first if model files do not exist.")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    st.error("Model not found. Run: python train.py --fast")
    st.stop()

meta = load_meta(META_PATH) or {}
thresholds = meta.get("thresholds", {label: 0.5 for label in LABELS})

for label in LABELS:
    thresholds[label] = st.sidebar.slider(
        f"Threshold: {label}",
        min_value=0.05,
        max_value=0.95,
        value=float(thresholds[label]),
        step=0.05,
    )

sample = st.text_area(
    "Enter comment",
    value="You are stupid and useless.",
    height=120,
)

if st.button("Analyze"):
    probs = np.asarray(model.predict_proba([sample]))[0]
    result = []
    for i, label in enumerate(LABELS):
        p = float(probs[i])
        detected = p >= thresholds[label]
        result.append(
            {
                "label": label,
                "probability": round(p, 4),
                "threshold": thresholds[label],
                "detected": detected,
            }
        )

    df = pd.DataFrame(result)
    st.dataframe(df, use_container_width=True)

    active = [r["label"] for r in result if r["detected"]]
    if active:
        st.warning("Detected labels: " + ", ".join(active))
    else:
        st.success("No labels detected at current thresholds.")
