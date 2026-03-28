#!/usr/bin/env bash
# =============================================================================
#  demo.sh  —  One-command verification for Toxic Comment Detector
#  Usage:    bash demo.sh
#            bash demo.sh --skip-train    (if model already trained)
# =============================================================================

set -e   # exit on any error

SKIP_TRAIN=false
for arg in "$@"; do
  [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

BOLD="\033[1m"; GREEN="\033[92m"; CYAN="\033[96m"
RED="\033[91m"; RESET="\033[0m"

banner() { echo -e "\n${CYAN}${BOLD}━━━ $1 ━━━${RESET}"; }
ok()     { echo -e "${GREEN}✓ $1${RESET}"; }
err()    { echo -e "${RED}✗ $1${RESET}"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   Toxic Comment Detector — Demo Verification ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Step 1: Check Python ──────────────────────────────────────────────────────
banner "Step 1/5  Python environment"
python --version || err "Python not found. Install Python 3.9+"
ok "Python OK"

# ── Step 2: Install dependencies ─────────────────────────────────────────────
banner "Step 2/5  Installing dependencies"
pip install -r requirements.txt -q && ok "Dependencies installed"

# ── Step 3: Fast training ─────────────────────────────────────────────────────
banner "Step 3/5  Training (FAST MODE — 1 epoch, 5000 rows)"
if [ "$SKIP_TRAIN" = true ]; then
    echo "  Skipping training (--skip-train flag set)"
    [[ -f "saved_model/model_weights.pt" ]] || err "No saved model found. Remove --skip-train to train."
    ok "Existing model found"
else
    python train.py --fast && ok "Training complete"
fi

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
banner "Step 4/5  Evaluation + threshold tuning"
python evaluate.py && ok "Evaluation complete"

# ── Step 5: Demo predictions ─────────────────────────────────────────────────
banner "Step 5/5  Demo predictions"
python predict.py --demo && ok "Demo predictions complete"

# ── Done ─────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}"
echo "══════════════════════════════════════════════"
echo "  All checks passed! ✓  Project is demo-ready"
echo "══════════════════════════════════════════════"
echo -e "${RESET}"
echo "  Now launch the UI:"
echo -e "  ${CYAN}streamlit run app.py${RESET}"
echo ""
echo "  Open → http://localhost:8501"
echo "  Go to the '💡 Demo ← Start here' tab for instant demo"
echo ""
