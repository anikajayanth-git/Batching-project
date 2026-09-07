import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

def inject_fraud(filepath="transactions.csv", output="transactions_with_fraud.csv"):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fraud_cases = []

    for i in range(20):
        fraud_cases.append({
            "transaction_id": f"FRD{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(8000, 15000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} 0{random.randint(1,4)}:{random.randint(0,59):02d}:00")
        })

    duplicates = df.sample(10).copy()
    fraud_cases.extend(duplicates.to_dict("records"))

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
            "timestamp": pd.Timestamp(f"2026-07-15 14:{i+1:02d}:00")
        })

    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"MIS{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(1000, 5000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} 23:{random.randint(0,59):02d}:00")
        })

    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"WKD{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(5000, 12000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.choice([5,6,12,13,19,20,26,27]):02d} {random.randint(8,18):02d}:00:00")
        })

    fraud_df = pd.DataFrame(fraud_cases)
    combined = pd.concat([df, fraud_df], ignore_index=True)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined.to_csv(output, index=False)

    print(f"Original transactions:  {len(df)}")
    print(f"Fraud cases injected:   {len(fraud_df)}")
    print(f"Total transactions:     {len(combined)}")
    print(f"\nFraud breakdown:")
    print(f"  Late-night large batch:     20")
    print(f"  Duplicate transactions:     10")
    print(f"  Structuring (rapid small):  15")
    print(f"  Reconciliation mismatches:  10")
    print(f"  Weekend batch anomalies:    10")
    print(f"\nSaved to {output}")

inject_fraud()
