# Bug Priority Scoring

A simple machine learning demo for prioritizing software bugs.

This project includes:
- a synthetic dataset generator
- a training script for a bug priority classifier
- a Streamlit web app for interactive prediction

## Problem
Given basic bug signals such as environment, affected users, crash impact, payment impact, workaround availability, and customer tier, predict whether a bug should be treated as **High Priority**.

## Project structure

```text
.
├── app.py
├── generate_data.py
├── train_model.py
├── data/
│   └── bugs.csv
├── models/
│   └── bug_priority_pipeline.joblib
├── requirements.txt
└── src/
    └── bug_priority/
        ├── __init__.py
        ├── data_generation.py
        └── model.py
```

## Quick start

### 1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate dataset
```bash
python generate_data.py
```

### 4. Train model
```bash
python train_model.py
```

### 5. Run Streamlit app
```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Demo inputs
Try these examples in the UI.

### Likely high priority
- environment: prod
- module: payments
- affected users: 500
- frequency last 24h: 1200
- crash: yes
- payment related: yes
- workaround: no
- customer tier: enterprise
- days since release: 1
- sla breach risk: yes

### Likely lower priority
- environment: dev
- module: ui
- affected users: 3
- frequency last 24h: 5
- crash: no
- payment related: no
- workaround: yes
- customer tier: free
- days since release: 45
- sla breach risk: no

## Notes
- The dataset is synthetic, designed for presentation/demo purposes.
- The model is a binary classifier: `high_priority` vs `not_high_priority`.
- The app also shows the predicted probability for easier explanation during a presentation.
