# Log Priority Scoring

A machine learning demo for prioritizing incidents from raw log strings.

This project follows a practical approach:
1. ingest raw log lines that look like real production logs
2. extract structured signals using regex + keyword rules
3. score each log with a machine learning model
4. rank logs from highest to lowest priority

## Project structure

```text
.
├── app.py
├── generate_data.py
├── train_model.py
├── data/
│   ├── demo_log_batch.txt
│   └── logs.csv
├── models/
│   └── log_priority_pipeline.joblib
├── requirements.txt
└── src/
    └── bug_priority/
        ├── __init__.py
        ├── data_generation.py
        ├── log_parser.py
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

### 3. Generate synthetic log dataset
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

## Demo pages

### 1. Single Log Demo
Paste one raw log line and click **Evaluate**.
The app will:
- parse the log with regex/rules
- extract structured features
- predict the priority score
- show the extracted signals used for scoring

### 2. Batch Log Ranking Demo
Paste multiple log lines, one per line, then click **Rank Logs**.
The app will:
- parse every log
- add repetition counts for similar patterns
- score each log
- sort from highest to lowest priority
- let you download the ranked result as CSV

Included sample input:
- `data/demo_log_batch.txt`

## Extracted features
The parser extracts practical signals such as:
- environment: `prod`, `staging`, `dev`
- level: `INFO`, `WARN`, `ERROR`, `FATAL`
- service name and logical module
- error code
- payment/auth/db/timeout/rate-limit/oom flags
- customer impact flag
- workaround flag
- SLA breach risk flag
- estimated affected users
- estimated frequency
- repeat count for the same pattern

## Notes
- The dataset is synthetic, but the raw logs are generated to look realistic enough for a presentation demo.
- The approach is intentionally practical: rule/regex feature extraction first, then ML scoring.
- This is closer to how many real systems work than sending raw logs directly into a model with no preprocessing.
