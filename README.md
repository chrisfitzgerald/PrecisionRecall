# Precision & Recall in TAR | Interactive Visualizer

An interactive Streamlit app for explaining precision, recall, and the precision-recall tradeoff in the context of Technology-Assisted Review (TAR) for legal e-discovery.

## What It Does

The app walks through five narrative sections:

1. **The Problem** — Why TAR exists and what precision/recall measure
2. **The Dataset** — 750 synthetic e-discovery documents (emails, memos, contracts, etc.) with a 20% responsiveness rate
3. **The Decision Boundary** — Interactive threshold slider with a live confusion matrix
4. **The Tradeoff** — Precision-Recall curve with a live operating point, plus F1 curve
5. **Practical Implications** — Three named review strategies with computed outcomes

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# (Already done) Generate the synthetic dataset
python data/generate_dataset.py

# Run the app
streamlit run app.py
```

## Deployment to Streamlit Community Cloud

1. Push this repo to GitHub (include `data/documents.csv`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select repo/branch → set main file to `app.py`
4. Click **Deploy**

No secrets or environment variables needed. The app is fully self-contained.

## Project Structure

```
├── app.py                    # Main app (5 narrative sections)
├── data/
│   ├── generate_dataset.py   # One-time data generation script
│   └── documents.csv         # Pre-generated synthetic dataset (750 docs)
├── components/
│   ├── confusion_matrix.py   # Plotly 2×2 confusion matrix figure
│   ├── pr_curve.py           # Plotly PR curve + F1 curve figures
│   └── metrics_cards.py      # Precision/Recall/F1 metric cards row
├── utils/
│   └── calculations.py       # Pure metric math (no Streamlit dependencies)
├── assets/
│   └── style.css             # Custom CSS
└── .streamlit/
    └── config.toml           # Theme configuration
```
