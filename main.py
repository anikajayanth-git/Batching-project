import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── CONSTANTS (from Federal Reserve Payments Study 2024) ──────────
ACH_AVG_AMOUNT = 2642
ACH_STD_DEV = 1500
FAILURE_RATE = 1.5
PENDING_RATE = 3.0
SUCCESS_RATE = 95.5

# ── STEP 1: GENERATE TRANSACTIONS ────────────────────────────────
def generate_transactions(n=100000):
    print("\n── Step 1: Generating Transactions ──────────────")
    transactions = []
    for i in range(n):
        payment_type = random.choices(
            ["batch", "real-time"], weights=[70, 30]
        )[0]
        amount = round(np.random.normal(ACH_AVG_AMOUNT, ACH_STD_DEV), 2)
        amount = max(1, amount)
        status = random.choices(
            ["completed", "failed", "pending"],
            weights=[SUCCESS_RATE, FAILURE_RATE, PENDING_RATE]
        )[0]
        transactions.append({
            "transaction_id": f"TXN{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": amount,
            "payment_type": payment_type,
            "status": status,
            "timestamp": fake.date_time_this_month(),
            "is_fraud": 0
        })
    df = pd.DataFrame(transactions)
    print(f"Generated {n} normal transactions")
    print(f"Batch: {(df['payment_type']=='batch').sum()} | Real-time: {(df['payment_type']=='real-time').sum()}")
    return df

# ── STEP 2: INJECT FRAUD ─────────────────────────────────────────
def inject_fraud(df):
    print("\n── Step 2: Injecting Fraud ───────────────────────")
    fraud_cases = []

    # Type 1: Late-night large batch
    for i in range(20):
        fraud_cases.append({
            "transaction_id": f"FRD{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(8000, 15000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} 0{random.randint(1,4)}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    # Type 2: Duplicates
    duplicates = df.sample(10).copy()
    duplicates["transaction_id"] = [f"DUP{i+1:04d}" for i in range(10)]
    duplicates["is_fraud"] = 1
    fraud_cases.extend(duplicates.to_dict("records"))

    # Type 3: Structuring
    base_sender = fake.name()
    base_receiver = fake.name()
    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"STR{i+1:04d}",
            "sender": base_sender,
            "receiver": base_receiver,
            "amount": round(random.uniform(900, 999), 2),
            "payment_type": "real-time",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-15 14:{i+1:02d}:00"),
            "is_fraud": 1
        })

    # Type 4: Reconciliation mismatches
    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"MIS{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(1000, 5000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} 23:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    # Type 5: Weekend batch
    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"WKD{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(5000, 12000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.choice([5,6,12,13,19,20,26,27]):02d} {random.randint(8,18):02d}:00:00"),
            "is_fraud": 1
        })

    # Type 6: Round numbers
    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"RND{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": float(random.choice([1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000])),
            "payment_type": random.choice(["batch", "real-time"]),
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    # Type 7: Velocity fraud
    compromised_sender = fake.name()
    base_time = pd.Timestamp("2026-07-20 03:00:00")
    for i in range(20):
        fraud_cases.append({
            "transaction_id": f"VEL{i+1:04d}",
            "sender": compromised_sender,
            "receiver": fake.name(),
            "amount": round(random.uniform(200, 2000), 2),
            "payment_type": "real-time",
            "status": "completed",
            "timestamp": base_time + timedelta(minutes=i*2),
            "is_fraud": 1
        })

    # Type 8: Ghost transactions
    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"GHT{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(0.01, 1.00), 2),
            "payment_type": "real-time",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    # Type 9: Batch to real-time switching
    batch_sample = df[df["payment_type"] == "batch"].sample(10).copy()
    batch_sample["transaction_id"] = [f"SWT{i+1:04d}" for i in range(10)]
    batch_sample["payment_type"] = "real-time"
    batch_sample["amount"] = batch_sample["amount"] * 1.5
    batch_sample["is_fraud"] = 1
    fraud_cases.extend(batch_sample.to_dict("records"))

    # Type 10: Failed marked completed
    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"FCM{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(500, 8000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} {random.randint(17,20):02d}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    fraud_df = pd.DataFrame(fraud_cases)
    fraud_df["is_fraud"] = 1
    df["is_fraud"] = 0
    combined = pd.concat([df, fraud_df], ignore_index=True)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    print(f"Fraud cases injected:  {len(fraud_df)}")
    print(f"Total transactions:    {len(combined)}")
    print(f"Fraud rate:            {len(fraud_df)/len(combined)*100:.1f}%")
    return combined

# ── STEP 3: ETL PIPELINE ─────────────────────────────────────────
def run_etl(df):
    print("\n── Step 3: Running ETL Pipeline ─────────────────")

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_late_night'] = df['hour'].between(0, 5).astype(int)
    df['is_large_amount'] = (df['amount'] > 5000).astype(int)
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    le = LabelEncoder()
    df['payment_type_encoded'] = le.fit_transform(df['payment_type'])

    # ML model
    features = ['amount', 'hour', 'day_of_week', 'is_weekend',
                'is_late_night', 'is_large_amount', 'amount_zscore',
                'payment_type_encoded']
    model = IsolationForest(contamination=0.12, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[features])
    df['is_anomaly'] = df['anomaly_score'] == -1

    print(f"Anomalies detected:    {df['is_anomaly'].sum()}")
    return df

# ── STEP 4: MEASURE ACCURACY ─────────────────────────────────────
def measure_accuracy(df):
    print("\n── Step 4: Model Accuracy ────────────────────────")
    cm = confusion_matrix(df['is_fraud'], df['is_anomaly'].astype(int))
    print(classification_report(
        df['is_fraud'],
        df['is_anomaly'].astype(int),
        target_names=['Normal', 'Fraud']
    ))
    print(f"True Negatives  (normal, correctly ignored): {cm[0][0]}")
    print(f"False Positives (normal, wrongly flagged):   {cm[0][1]}")
    print(f"False Negatives (fraud, missed by model):    {cm[1][0]}")
    print(f"True Positives  (fraud, correctly caught):   {cm[1][1]}")
    print(f"\nFraud catch rate: {cm[1][1]}/{cm[1][0]+cm[1][1]} = {cm[1][1]/(cm[1][0]+cm[1][1])*100:.1f}%")

# ── RUN EVERYTHING ────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_transactions(n=100000)
    df = inject_fraud(df)
    df = run_etl(df)
    df.to_csv("transactions_final.csv", index=False)
    measure_accuracy(df)
    print("\n── Complete! Saved to transactions_final.csv ────")