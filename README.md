# BatchBridge — ACH Batch & Real-Time Payment Reconciliation System

An end-to-end fraud-detection pipeline that simulates ACH (Automated Clearing House) payment
data, injects realistic fraud patterns, and uses unsupervised anomaly detection to flag
suspicious transactions — visualized through an interactive Streamlit dashboard.

## Project Overview

Banks and financial institutions process both batch and real-time ACH payments, each with
different fraud risk profiles. This project simulates that environment end-to-end: generating
realistic transaction volumes calibrated to Federal Reserve statistics, injecting known fraud
patterns, and building an Isolation Forest model to detect anomalies — without ever telling the
model which transactions are fraudulent during training (a realistic constraint, since most
fraud is unlabeled at detection time).

## Pipeline Steps

1. **Generate Transactions** (`generate_transactions.py`) — simulates 100,000+ ACH transactions,
   with amounts and batch/real-time split calibrated to Federal Reserve Payments Study statistics
   (average transaction ~$2,642, 95.5% success rate)
2. **Inject Fraud** (`inject_fraud.py`) — injects 10 distinct fraud patterns into the transaction
   set, each modeled on a real banking fraud typology (see below)
3. **ETL & Feature Engineering** (`etl_pipeline.py`) — extracts, transforms, and enriches
   transaction data with 12+ engineered features (time-of-day, weekend flags, z-scored amounts,
   sender transaction frequency, round-number detection, etc.)
4. **Anomaly Detection** — applies an Isolation Forest model (unsupervised) to flag suspicious
   transactions based on engineered features alone
5. **Evaluation** — measures precision, recall, and fraud catch rate against the known injected
   fraud labels
6. **Dashboard** (`dashboard.py`) — interactive Streamlit app for exploring results, adjusting
   model sensitivity, and visualizing fraud detection performance in real time

## Fraud Patterns Simulated

| Pattern | Description |
|---|---|
| Late-night large batch | High-value batch payments processed in early morning hours |
| Duplicate transactions | Same transaction processed twice (double billing) |
| Structuring | Repeated payments just under $1,000 between the same parties, to evade reporting thresholds |
| Reconciliation mismatches | Transactions marked "completed" that likely never settled |
| Weekend batch anomalies | Batch payments processed on weekends, when batch systems don't normally run |
| Round-number transactions | Suspiciously round amounts (e.g., exactly $10,000), atypical of real payments |
| Velocity fraud | One sender issuing many rapid transactions in a short window (account takeover pattern) |
| Ghost transactions | Tiny test transactions used to verify an account is active before a larger attack |
| Batch-to-real-time switching | A payment reappearing under a different payment type — signals system manipulation |
| Failed-but-marked-completed | Transactions the system reports as completed despite failing to settle |

## Results

- Simulated and processed 100,000+ transactions with 145 injected fraud cases across 10 fraud types
- Engineered 12+ features and applied Isolation Forest anomaly detection (contamination rate
  tuned to the actual fraud rate in the dataset)
- Interactive dashboard reports precision, recall, and F1 score in real time, along with a
  sensitivity analysis showing the tradeoff between fraud catch rate and false alarm rate as
  model sensitivity is adjusted
- Per-fraud-type breakdown shows detection rate varies by pattern — some fraud types (e.g.,
  large late-night batches) are easier for the model to catch than subtler ones (e.g.,
  structuring)

## Files

- `main.py` — runs the full pipeline end-to-end (generate → inject fraud → ETL → detect →
  evaluate), the primary entry point for reproducing results
- `dashboard.py` — interactive Streamlit dashboard with adjustable simulation controls,
  visualizations, and model performance metrics
- `generate_transactions.py` — standalone transaction generator
- `inject_fraud.py` — standalone fraud injection script (10 fraud patterns)
- `etl_pipeline.py` — standalone ETL and feature engineering pipeline
- `load_fed_data.py` — loads and inspects Federal Reserve Payments Study data used to calibrate
  transaction statistics
- `transactions.csv`, `transactions_with_fraud.csv`, `transactions_enriched.csv`,
  `transactions_final.csv` — data at various pipeline stages

## Tools & Libraries

Python, Pandas, NumPy, Scikit-learn (Isolation Forest), Faker, Streamlit, Matplotlib

## How to Run

```bash
pip install -r requirements.txt
python main.py          # runs the full pipeline and prints evaluation metrics
streamlit run dashboard.py   # launches the interactive dashboard
```
