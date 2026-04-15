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
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows CMD:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
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

If `streamlit` is not in PATH, use:
```bash
python -m streamlit run app.py
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

## Batch Ranking Demo
The Streamlit app now includes two pages:
- **Single Bug Demo**: predict the priority for one bug interactively
- **Batch Ranking Demo**: upload a CSV, score all bugs, sort them from highest to lowest priority, and download the ranked output

Required columns for batch upload:
- `environment`
- `module`
- `customer_tier`
- `affected_users_count`
- `frequency_last_24h`
- `days_since_release`
- `is_crash`
- `is_payment_related`
- `has_workaround`
- `sla_breach_risk`

Optional column:
- `bug_id` for easier reading in the ranked result

## Notes
- The dataset is synthetic, designed for presentation/demo purposes.
- The model is a binary classifier: `high_priority` vs `not_high_priority`.
- The app also shows the predicted probability for easier explanation during a presentation.
- For realistic usage, the batch page is closer to how a triage system would be used in practice.
