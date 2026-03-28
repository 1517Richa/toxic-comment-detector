import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import LABELS, load_meta

MODEL_PATH = "saved_model/model.joblib"
META_PATH = "saved_model/meta.json"

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="shield",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --bg: #f6f7f9;
        --ink: #11212d;
        --muted: #5d6c77;
        --brand: #0f766e;
        --brand-2: #f59e0b;
        --soft: #ffffff;
        --ok: #15803d;
        --bad: #b91c1c;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at -10% -20%, #d6f7f0 0%, transparent 60%),
            radial-gradient(900px 500px at 110% -10%, #ffe9c7 0%, transparent 55%),
            var(--bg);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14313d 0%, #1f4b5d 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #edf6ff !important;
    }

    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #155e75 50%, #0f766e 100%);
        border-radius: 18px;
        padding: 28px 30px;
        color: #f8fffe;
        box-shadow: 0 12px 40px rgba(12, 32, 46, 0.20);
        margin-bottom: 14px;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.3px;
    }
    .hero p {
        margin: 6px 0 0 0;
        color: #dff8ff;
        font-size: 1rem;
    }

    .glass {
        background: rgba(255, 255, 255, 0.74);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 10px 24px rgba(17, 33, 45, 0.08);
    }

    .kpi {
        background: var(--soft);
        border: 1px solid #dce9ef;
        border-radius: 14px;
        padding: 10px 12px;
        text-align: center;
    }
    .kpi .n {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: var(--brand);
        font-weight: 600;
    }
    .kpi .t {
        color: var(--muted);
        font-size: 0.82rem;
    }

    .label-card {
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border: 1px solid #d6e0e8;
        background: #ffffff;
    }
    .label-card.detected {
        border-color: #fecaca;
        background: #fff5f5;
    }
    .label-card.clean {
        border-color: #bbf7d0;
        background: #f3fff7;
    }
    .label-name {
        font-weight: 700;
        color: #0c2a36;
        text-transform: capitalize;
    }
    .verdict-bad { color: var(--bad); font-weight: 700; }
    .verdict-good { color: var(--ok); font-weight: 700; }

    .help-note {
        color: #3e4e59;
        font-size: 0.92rem;
        padding: 8px 2px 2px 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Toxic Comment Detector</h1>
      <p>Analyze text across six toxicity labels with adjustable sensitivity controls.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    st.error("Model not found. Run: python train.py --fast")
    st.stop()

meta = load_meta(META_PATH) or {}
thresholds = meta.get("thresholds", {label: 0.5 for label in LABELS})

st.sidebar.markdown("## Detection Sensitivity")
st.sidebar.caption("Lower threshold means more sensitive detection.")

for label in LABELS:
    thresholds[label] = st.sidebar.slider(
        f"Threshold: {label}",
        min_value=0.05,
        max_value=0.95,
        value=float(thresholds[label]),
        step=0.05,
    )

examples = {
    "Hostile": "You are stupid and useless.",
    "Neutral": "I appreciate your detailed feedback. Thank you.",
    "Threatening": "I will hurt you if you keep doing that.",
}

top_left, top_right = st.columns([1.6, 1], gap="large")

with top_left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    example_choice = st.selectbox("Quick examples", list(examples.keys()), index=0)
    sample = st.text_area(
        "Comment text",
        value=examples[example_choice],
        height=140,
        placeholder="Type any sentence or paste a comment here...",
    )
    analyze = st.button("Analyze Comment", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="kpi"><div class="n">{len(LABELS)}</div><div class="t">Labels</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        avg_t = sum(float(thresholds[l]) for l in LABELS) / len(LABELS)
        st.markdown(
            f'<div class="kpi"><div class="n">{avg_t:.2f}</div><div class="t">Avg Threshold</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="kpi"><div class="n">Live</div><div class="t">Inference</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="help-note">Tip: Tune thresholds in the sidebar to reduce or increase alert sensitivity per label.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

if analyze:
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
    st.markdown("### Results")
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("### Label Breakdown")
    for row in result:
        label_class = "detected" if row["detected"] else "clean"
        verdict_class = "verdict-bad" if row["detected"] else "verdict-good"
        verdict_text = "DETECTED" if row["detected"] else "CLEAN"
        st.markdown(
            (
                f'<div class="label-card {label_class}">'
                f'<span class="label-name">{row["label"].replace("_", " ")}</span><br>'
                f'Probability: <b>{row["probability"] * 100:.2f}%</b> | '
                f'Threshold: <b>{row["threshold"]:.2f}</b> | '
                f'Verdict: <span class="{verdict_class}">{verdict_text}</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    active = [r["label"] for r in result if r["detected"]]
    if active:
        st.warning("Detected labels: " + ", ".join(active))
    else:
        st.success("No labels detected at current thresholds.")
else:
    st.info("Enter text and click Analyze Comment to view predictions.")
