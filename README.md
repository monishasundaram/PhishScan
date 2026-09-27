# AI-Based Phishing URL Detection and Risk Analysis System

A working Flask + Scikit-learn/XGBoost app that takes a URL, extracts lexical
and (best-effort) domain-reputation features, classifies it as
**legitimate** or **phishing**, and shows an explainable risk score with the
specific signals that drove the decision.

## What's inside
- `features.py` — Tier-1 (lexical, instant) + Tier-2 (WHOIS/SSL, best-effort) feature extraction
- `train_model.py` — builds a demo training dataset, trains & compares Logistic Regression / Random Forest / XGBoost, recall-tunes the decision threshold, saves `data/model.joblib`
- `app.py` — Flask API + dashboard server, SQLite storage
- `templates/index.html` — Bootstrap + Chart.js dashboard (URL input → risk gauge → reasons → history → time-series chart)
- `Dockerfile` / `docker-compose.yml` — containerized deployment

## ⚠️ About the training data
This project ships with a **programmatically generated demo dataset**
(`train_model.py:build_dataset()`) combining real legitimate domains with
realistically-patterned synthetic phishing URLs, so it trains and runs
completely offline with no external downloads required. It's meant to prove
out the full pipeline (feature extraction → training → recall tuning →
serving → explainability), not as a production-grade phishing classifier.

### Training on real data
`train_model.py` now accepts real data directly:

```bash
python train_model.py --phishing-csv data/real_phishing.csv --legit-csv data/real_tranco.csv
```

- **`--phishing-csv`**: a [PhishTank](https://phishtank.org/) or
  [OpenPhish](https://openphish.com/) export. Any CSV with a `url` column
  works (PhishTank's format), as does a plain text file with one URL per
  line (OpenPhish's free feed format).
- **`--legit-csv`**: a [Tranco](https://tranco-list.eu/) top-sites export
  (`rank,domain` CSV, no header). Each bare domain is expanded into a few
  realistic URLs automatically.

The hand-crafted "hard" examples (typosquats, brand name stuffed into an
unrelated domain's path, real-brand login pages, brand mentions on news
sites) are always layered on top of whichever base dataset you choose —
they target specific weaknesses this project has already hit in testing,
independent of where the bulk data comes from.

Without either flag, it falls back to the synthetic demo dataset.

## Quick start (local)

```bash
pip install -r requirements.txt
python train_model.py        # trains model, writes data/model.joblib
python app.py                 # runs dev server on http://localhost:5000
```

Open http://localhost:5000, enter a URL, click **Analyze**.

## Quick start (Docker)

```bash
docker compose up --build
```

Visit http://localhost:5000. The model is trained at image-build time; the
`data/` volume persists the SQLite scan history and model across restarts.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/scan` | `{"url": "...", "deep_scan": true}` → prediction, risk score, reasons, SHAP feature breakdown |
| GET | `/api/v1/scans?limit=25` | Recent scan history |
| GET | `/api/v1/stats/timeseries` | Daily phishing/legitimate counts for charts |
| POST | `/api/v1/feedback` | `{"scan_id": 1, "user_label": "phishing"}` — user correction, for future retraining |
| GET | `/api/v1/health` | Model name + active decision threshold |

## How recall is prioritized

1. `train_model.py` trains all three models with `class_weight="balanced"`
   (or `scale_pos_weight` for XGBoost) so the *model itself* leans toward
   catching phishing, not just the decision threshold.
2. Model selection picks the best **F2-score** (recall weighted 2x precision)
   among Logistic Regression / Random Forest / XGBoost.
3. The decision threshold is then swept via the precision-recall curve to
   the lowest value that still achieves **recall ≥ 0.95** on the held-out
   test set — this threshold (not 0.5) is what `app.py` uses at inference
   time, and is reported by `/api/v1/health`.

## Explainability

Two layers of explanation ship together:
- **Rule-based reasons** (`app.py:REASON_RULES` + the brand/typosquat/keyword
  matchers) — clear, literal statements like "Domain closely resembles brand
  'paypal'". These are always available and never require the model.
- **SHAP feature contributions** — the top 5 features that actually moved
  *this specific prediction's* score, with direction (increases/decreases
  risk) and relative magnitude, computed via `shap.TreeExplainer` (Random
  Forest/XGBoost) or `shap.LinearExplainer` (Logistic Regression) against a
  100-row background sample saved alongside the model. Shown as a bar panel
  under "Model feature contributions" in the dashboard. Adds ~50-70ms to a
  scan; falls back to `[]` gracefully if SHAP fails for any reason.

## Notes on Tier-2 features in sandboxed/offline environments
`extract_tier2_features()` (WHOIS domain age, live SSL check) requires
outbound network access to arbitrary hosts. If that's unavailable (firewalled
server, CI sandbox, etc.) it fails gracefully to neutral values (`-1`) rather
than crashing the request — toggle "Deep scan" off in the UI, or pass
`"deep_scan": false` to `/api/v1/scan`, to skip these lookups entirely and
rely on Tier-1 lexical features only.

## Extending this project
- Swap in real PhishTank/OpenPhish + Tranco data (see above)
- Add SHAP-based per-prediction explanations (swap the rule-based `REASON_RULES` in `app.py`)
- Migrate SQLite → Postgres/MySQL for multi-instance deployment
- Add a scheduled retraining job that folds in `/api/v1/feedback` corrections
- Add typosquat/brand-similarity detection (Levenshtein distance to known brand domains)
