import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def extract(filepath="transactions.csv"):
    print("Extracting data...")
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} transactions")
    return df

def transform(df):
    print("Transforming data...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_late_night'] = df['hour'].between(0, 5).astype(int)
    df['is_large_amount'] = (df['amount'] > 5000).astype(int)
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    le = LabelEncoder()
    df['payment_type_encoded'] = le.fit_transform(df['payment_type'])
    features = ['amount', 'hour', 'day_of_week', 'is_weekend',
                'is_late_night', 'is_large_amount', 'amount_zscore',
                'payment_type_encoded']
    model = IsolationForest(contamination=0.015, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[features])
    df['is_anomaly'] = df['anomaly_score'] == -1
    print(f"Anomalies detected: {df['is_anomaly'].sum()}")
    return df

def load(df, output_path="transactions_enriched.csv"):
    print("Loading enriched data...")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print("\n── ETL Summary ──────────────────")
    print(f"Total transactions:  {len(df)}")
    print(f"Anomalies flagged:   {df['is_anomaly'].sum()}")
    print(f"Failed transactions: {(df['status'] == 'failed').sum()}")
    print(f"Pending:             {(df['status'] == 'pending').sum()}")
    print(f"Batch payments:      {(df['payment_type'] == 'batch').sum()}")
    print(f"Real-time payments:  {(df['payment_type'] == 'real-time').sum()}")
    print("─────────────────────────────────")

if __name__ == "main":
    df = extract()
    df = transform(df)
    load(df)

df = extract()
df = transform(df)
load(df)