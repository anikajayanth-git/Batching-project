import pandas as pd
import numpy as np
from faker import Faker
import random

fake=Faker()

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
    duplicates["transaction_id"] = duplicates["transaction_id"]
    duplicates["amount"] = duplicates["amount"]
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
    combined = combined.sample(frac=1).reset_index(drop=True)  # shuffle
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

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

def inject_fraud(filepath="transactions.csv", output="transactions_with_fraud.csv"):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fraud_cases = []

    # ── FRAUD TYPE 1: Late-night large batch payments
    # Signals: high amount + batch + late night
    for i in range(20):
        fraud_cases.append({
            "transaction_id": f"FRD{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(8000, 15000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(
                f"2026-07-{random.randint(1,28):02d} "
                f"0{random.randint(1,4)}:{random.randint(0,59):02d}:00"
            )
        })

    # ── FRAUD TYPE 2: Duplicate transactions
    # Same transaction processed twice — double billing
    duplicates = df.sample(10).copy()
    duplicates["transaction_id"] = [f"DUP{i+1:04d}" for i in range(10)]
    fraud_cases.extend(duplicates.to_dict("records"))

    # ── FRAUD TYPE 3: Structuring — breaking large payments into small ones
    # Same sender → same receiver, just under $1000 repeatedly
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

    # ── FRAUD TYPE 4: Reconciliation mismatches
    # Shows completed but happened at end of day — likely never settled
    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"MIS{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(1000, 5000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(
                f"2026-07-{random.randint(1,28):02d} "
                f"23:{random.randint(0,59):02d}:00"
            )
        })

    # ── FRAUD TYPE 5: Weekend large batch payments
    # Batch systems don't run on weekends
    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"WKD{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(5000, 12000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(
                f"2026-07-{random.choice([5,6,12,13,19,20,26,27]):02d} "
                f"{random.randint(8,18):02d}:00:00"
            )
        })

    # ── FRAUD TYPE 6: Round number transactions
    # Real payments are rarely exactly $5000 or $10000
    # Round numbers signal manual fraud entry
    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"RND{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": float(random.choice([
                1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000
            ])),
            "payment_type": random.choice(["batch", "real-time"]),
            "status": "completed",
            "timestamp": pd.Timestamp(
                f"2026-07-{random.randint(1,28):02d} "
                f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:00"
            )
        })

    # ── FRAUD TYPE 7: Velocity fraud
    # One sender blasting many payments in a short window
    # Classic account takeover pattern
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
            "timestamp": base_time + timedelta(minutes=i*2)
        })

    # ── FRAUD TYPE 8: Ghost transactions
    # Tiny amounts that fly under the radar
    # Used to test if an account is active before a big hit
    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"GHT{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(0.01, 1.00), 2),
            "payment_type": "real-time",
            "status": "completed",
            "timestamp": pd.Timestamp(
                f"2026-07-{random.randint(1,28):02d} "
                f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:00"
            )
        })

    # ── FRAUD TYPE 9: Batch-to-real-time switching
    # Payment starts as batch then reappears as real-time
    # Signals system manipulation
    batch_sample = df[df["payment_type"] == "batch"].sample(10).copy()
    batch_sample["transaction_id"] = [f"SWT{i+1:04d}" for i in range(10)]
    batch_sample["payment_type"] = "real-time"
    batch_sample["amount"] = batch_sample["amount"] * 1.5
    fraud_cases.extend(batch_sample.to_dict("records"))

    # ── FRAUD TYPE 10: Failed but marked completed
    # Reconciliation nightmare — system says done but money never moved
    for i in range(10):
        fraud_cases.append({
            "transaction_id": f"FCM{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": round(random.uniform(500, 8000), 2),
            "payment_type": "batch",
            "status": "completed",
            "timestamp": pd.Timestamp(
                f"2026-07-{random.randint(1,28):02d} "
                f"{random.randint(17,20):02d}:{random.randint(0,59):02d}:00"
            )
        })

    # combine and shuffle
    fraud_df = pd.DataFrame(fraud_cases)
    fraud_df["is_fraud"] = 1
    df["is_fraud"] = 0
    combined = pd.concat([df, fraud_df], ignore_index=True)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined.to_csv(output, index=False)

    print(f"── Fraud Injection Summary ───────────────")
    print(f"Original transactions:        {len(df)}")
    print(f"Fraud cases injected:         {len(fraud_df)}")
    print(f"Total transactions:           {len(combined)}")
    print(f"Fraud rate:                   {len(fraud_df)/len(combined)*100:.1f}%")
    print(f"\nFraud type breakdown:")
    print(f"  Late-night large batch:     20")
    print(f"  Duplicate transactions:     10")
    print(f"  Structuring:                15")
    print(f"  Reconciliation mismatches:  10")
    print(f"  Weekend batch anomalies:    10")
    print(f"  Round number transactions:  15")
    print(f"  Velocity fraud:             20")
    print(f"  Ghost transactions:         15")
    print(f"  Batch-to-realtime switch:   10")
    print(f"  Failed marked completed:    10")
    print(f"──────────────────────────────────────────")
    print(f"\nSaved to {output}")

inject_fraud()
