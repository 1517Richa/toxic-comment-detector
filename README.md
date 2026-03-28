<div align="center">

# 🛡️ Toxic Comment Detector

### Multi-label toxicity classification using BERT and Hugging Face Transformers

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.38%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

[Live Demo](https://YOUR-APP-NAME.streamlit.app) · [Quick Start](#-quick-start) · [Documentation](#-project-structure)

</div>

---

## 📌 What This Project Does

Online platforms face a constant challenge: detecting harmful comments at scale. This project fine-tunes **BERT** (`bert-base-uncased`) on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset to classify comments into **6 toxicity categories simultaneously**.

Given a comment like:

> *"You are such an absolute idiot. I can't believe how stupid you are."*

The model outputs a confidence score **per label**:

| Label | Score | Decision |
|---|---|---|
| toxic | 91.2% | ✗ DETECTED |
| severe_toxic | 14.5% | ✓ clean |
| obscene | 8.1% | ✓ clean |
| threat | 3.2% | ✓ clean |
| insult | 78.3% | ✗ DETECTED |
| identity_hate | 2.0% | ✓ clean |

---

## ✨ Features

- **Multi-label classification** — 6 independent binary labels per comment
- **BERT fine-tuning** — `bert-base-uncased` with AdamW + linear warmup
- **Per-label threshold tuning** — each label gets its own optimal decision threshold
- **Early stopping** — stops training when validation F1 stops improving
- **Best-model checkpointing** — automatically saves the highest-performing checkpoint
- **Full evaluation suite** — per-label F1, confusion matrices, macro/micro metrics
- **Interactive Streamlit UI** — colour-coded verdicts, probability bars, batch analysis
- **Fast demo mode** — `--fast` flag trains in ~5 minutes for presentations
- **GPU-ready** — automatically uses CUDA when available

---

## 🏗️ Architecture

```
Input Comment
      │
      ▼  BertTokenizer (WordPiece, max_len=128)
      │
      ▼  bert-base-uncased
         12 layers · 768 hidden · 12 attention heads
      │
      ▼  [CLS] pooler_output  (768-dim)
      │
      ▼  Dropout(p=0.3)
      │
      ▼  Linear(768 → 6)    ← one logit per label
      │
      ▼  Sigmoid            ← probability per label
      │
      ▼  Per-label threshold → binary prediction
```

| Component | Detail |
|---|---|
| Loss | `BCEWithLogitsLoss` (binary CE per label) |
| Optimizer | `AdamW` with weight decay (biases excluded) |
| Scheduler | Linear warmup (10%) → linear decay |
| Gradient clipping | max_norm = 1.0 |
| Regularisation | Dropout(0.3) on [CLS] representation |

---

## 🏷️ Toxicity Labels

| Label | Description |
|---|---|
| `toxic` | Generally rude or disrespectful |
| `severe_toxic` | Very hateful or aggressive content |
| `obscene` | Sexual or vulgar language |
| `threat` | Threats toward a person |
| `insult` | Insulting or demeaning content |
| `identity_hate` | Hatred toward a group or identity |

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Model | `bert-base-uncased` via Hugging Face |
| Training | PyTorch 2.1+ |
| Tokeniser | `AutoTokenizer` (Hugging Face Transformers) |
| Evaluation | scikit-learn |
| Visualisation | matplotlib |
| Web UI | Streamlit |
| Data | pandas, numpy |

---

## 📂 Project Structure

```
toxic-comment-detector/
│
├── utils.py          # Shared config, model class, Dataset, helpers
├── train.py          # Training loop with early stopping + checkpointing
├── evaluate.py       # Per-label evaluation, confusion matrices, threshold tuning
├── predict.py        # Inference — single / batch / demo CLI
├── app.py            # Streamlit web UI
├── demo.sh           # One-command setup and verification script
├── requirements.txt  # Python dependencies
├── README.md
│
├── data/
│   └── README.md     # Instructions for downloading the Jigsaw dataset
│
└── saved_model/
    └── README.md     # Notes on model files (weights excluded from repo)
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/toxic-comment-detector.git
cd toxic-comment-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the dataset *(optional)*

Download `train.csv` from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it at `data/train.csv`.

> **No Kaggle account?** Skip this step — the project auto-generates synthetic data for quick testing.

### 5. Train

```bash
# Full training
python train.py

# ⚡ Fast mode — 1 epoch, 5000 rows (~5 min on CPU, great for demos)
python train.py --fast
```

### 6. Evaluate

```bash
python evaluate.py
```

### 7. Predict

```bash
# Preloaded demo examples — no typing needed
python predict.py --demo

# Single comment
python predict.py --text "You are such an idiot!"

# Interactive
python predict.py
```

### 8. Launch the UI

```bash
streamlit run app.py
# → open http://localhost:8501
```

### One-command verification

```bash
bash demo.sh
```

---

## 🚀 Live Demo on GitHub (Streamlit Cloud)

You can publish a live app directly from this GitHub repository.

### Steps

1. Push latest code to your `main` branch.
2. Go to: https://share.streamlit.io
3. Click **New app** and select:
  - **Repository**: `1517Richa/toxic-comment-detector`
  - **Branch**: `main`
  - **Main file path**: `app.py`
4. Click **Deploy**.

### Notes

- The app now auto-creates a lightweight demo model on first run if model files are missing.
- `runtime.txt` is included so Streamlit Cloud uses a compatible Python runtime.
- After deploy, replace `https://YOUR-APP-NAME.streamlit.app` at the top of this README with your real app URL.

---

## 🌐 Streamlit UI

The app has **5 tabs**:

| Tab | Description |
|---|---|
| 💡 **Demo ← Start here** | 6 preloaded examples, one-click analysis |
| 🔍 **Analyse** | Type any comment → coloured verdict + probability bars |
| 📋 **Batch** | Paste or upload comments → download CSV results |
| 📊 **Confusion Matrix** | Per-label heatmaps + tuned thresholds table |
| ℹ️ **About** | Architecture diagram + tech stack |

The **threshold sliders** in the sidebar let you adjust detection sensitivity per label in real time.

> **Screenshot placeholder** — add UI screenshots here before submission

---

## ☁️ Google Colab

Run the entire project in the cloud with no local setup:

```python
# Step 1 — Clone the repo
!git clone https://github.com/YOUR_USERNAME/toxic-comment-detector.git
%cd toxic-comment-detector

# Step 2 — Install dependencies
!pip install -r requirements.txt -q

# Step 3 — Train (fast mode)
!python train.py --fast

# Step 4 — Evaluate
!python evaluate.py

# Step 5 — Run predictions
!python predict.py --demo
```

For the Streamlit UI in Colab, use [ngrok](https://ngrok.com/) or LocalTunnel:

```python
!pip install pyngrok -q
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 📊 Sample Training Output

```
──────────────────────────────────────────────────────────────
  Epoch  1   elapsed 47.3s
──────────────────────────────────────────────────────────────
  train_loss         0.1842
  val_loss           0.1531
  val_macro_f1       0.7214
  val_micro_f1       0.8103
  val_precision      0.7420
  val_recall         0.7012

  Per-label F1:
    toxic              ████████████████     0.842
    severe_toxic       ████████             0.412
    obscene            ███████████████      0.782
    threat             ████                 0.213
    insult             ██████████████       0.731
    identity_hate      ████                 0.223
──────────────────────────────────────────────────────────────
  ✓ Best model saved  (val_f1=0.7214)
```

---

## 📈 Results *(Jigsaw full dataset · 3 epochs · GPU)*

| Label | Precision | Recall | F1 |
|---|---|---|---|
| toxic | 0.94 | 0.95 | 0.94 |
| severe_toxic | 0.88 | 0.71 | 0.79 |
| obscene | 0.96 | 0.95 | 0.96 |
| threat | 0.91 | 0.65 | 0.76 |
| insult | 0.93 | 0.92 | 0.93 |
| identity_hate | 0.90 | 0.71 | 0.79 |
| **Macro avg** | **0.92** | **0.82** | **0.86** |

---

## ⚙️ Configuration

All hyperparameters are in `CONFIG` inside `utils.py`:

```python
CONFIG = {
    "model_name":        "bert-base-uncased",   # or "unitary/toxic-bert"
    "max_len":           128,
    "epochs":            5,
    "batch_size":        16,
    "learning_rate":     2e-5,
    "warmup_ratio":      0.10,
    "weight_decay":      0.01,
    "dropout":           0.30,
    "patience":          2,       # early stopping
    "monitor":           "val_f1",
}
```

**Use a domain-pretrained model (bonus):**

```bash
python train.py --model unitary/toxic-bert --fast
```

---

## 🔄 After Cloning — Verify Everything Works

```bash
bash demo.sh
# Runs: install → train (fast) → evaluate → predict → prints success
```

Expected final output:
```
══════════════════════════════════════════════
  All checks passed! ✓  Project is demo-ready
══════════════════════════════════════════════
  Now launch the UI:
  streamlit run app.py
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
The Jigsaw Toxic Comment dataset is subject to [Kaggle's competition rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules).

---

<div align="center">

Made with ❤️ using BERT · PyTorch · Hugging Face · Streamlit

</div>
