import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="BatchBridge — ACH Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# ── HEADER ────────────────────────────────────────────────────────
st.title("🏦 BatchBridge")
st.subheader("ACH Batch Payment Fraud Detection System")
st.caption("Simulating real-world batch vs real-time payment reconciliation using Federal Reserve data")

st.divider()

# ── SIDEBAR CONTROLS ──────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Controls")
n_transactions = st.sidebar.slider("Number of transactions", 500, 5000, 1000, 100)
contamination = st.sidebar.slider("Model sensitivity", 0.05, 0.20, 0.12, 0.01)
show_raw_data = st.sidebar.checkbox("Show raw transaction data", False)

# ── GENERATE DATA ─────────────────────────────────────────────────
@st.cache_data
def run_pipeline(n, contamination_rate):

    # Fed constants
    ACH_AVG = 2642
    ACH_STD = 1500

    # Generate normal transactions
    transactions = []
    for i in range(n):
        payment_type = random.choices(["batch", "real-time"], weights=[70, 30])[0]
        amount = max(1, round(np.random.normal(ACH_AVG, ACH_STD), 2))
        status = random.choices(
            ["completed", "failed", "pending"],
            weights=[95.5, 1.5, 3.0]
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

    # Inject fraud
    fraud_cases = []

    for i in range(20):
        fraud_cases.append({
            "transaction_id": f"FRD{i+1:04d}",
            "sender": fake.name(), "receiver": fake.name(),
            "amount": round(random.uniform(8000, 15000), 2),
            "payment_type": "batch", "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} 0{random.randint(1,4)}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    base_sender = fake.name()
    base_receiver = fake.name()
    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"STR{i+1:04d}",
            "sender": base_sender, "receiver": base_receiver,
            "amount": round(random.uniform(900, 999), 2),
            "payment_type": "real-time", "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-15 14:{i+1:02d}:00"),
            "is_fraud": 1
        })

    compromised = fake.name()
    base_time = pd.Timestamp("2026-07-20 03:00:00")
    for i in range(20):
        fraud_cases.append({
            "transaction_id": f"VEL{i+1:04d}",
            "sender": compromised, "receiver": fake.name(),
            "amount": round(random.uniform(200, 2000), 2),
            "payment_type": "real-time", "status": "completed",
            "timestamp": base_time + timedelta(minutes=i*2),
            "is_fraud": 1
        })

    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"GHT{i+1:04d}",
            "sender": fake.name(), "receiver": fake.name(),
            "amount": round(random.uniform(0.01, 1.00), 2),
            "payment_type": "real-time", "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    for i in range(15):
        fraud_cases.append({
            "transaction_id": f"RND{i+1:04d}",
            "sender": fake.name(), "receiver": fake.name(),
            "amount": float(random.choice([1000, 2000, 5000, 10000, 15000, 20000])),
            "payment_type": random.choice(["batch", "real-time"]),
            "status": "completed",
            "timestamp": pd.Timestamp(f"2026-07-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00"),
            "is_fraud": 1
        })

    fraud_df = pd.DataFrame(fraud_cases)
    combined = pd.concat([df, fraud_df], ignore_index=True)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    # ETL transform
    combined['hour'] = combined['timestamp'].dt.hour
    combined['day_of_week'] = combined['timestamp'].dt.dayofweek
    combined['is_weekend'] = combined['day_of_week'].isin([5, 6]).astype(int)
    combined['is_late_night'] = combined['hour'].between(0, 5).astype(int)
    combined['is_large_amount'] = (combined['amount'] > 5000).astype(int)
    combined['amount_zscore'] = (combined['amount'] - combined['amount'].mean()) / combined['amount'].std()
    le = LabelEncoder()
    combined['payment_type_encoded'] = le.fit_transform(combined['payment_type'])
    sender_avg = combined.groupby('sender')['amount'].transform('mean')
    combined['amount_vs_sender_avg'] = combined['amount'] / sender_avg
    combined['sender_tx_count'] = combined.groupby('sender')['transaction_id'].transform('count')
    combined['is_round_number'] = (combined['amount'] % 1000 == 0).astype(int)
    combined['is_tiny_amount'] = (combined['amount'] < 2).astype(int)
    actual_fraud_rate = combined['is_fraud'].mean()

    features = ['amount', 'hour', 'day_of_week', 'is_weekend',
                'is_late_night', 'is_large_amount', 'amount_zscore',
                'payment_type_encoded', 'amount_vs_sender_avg',
                'sender_tx_count', 'is_round_number', 'is_tiny_amount']

    model = IsolationForest(contamination=actual_fraud_rate, random_state=42)
    combined['anomaly_score'] = model.fit_predict(combined[features])
    combined['is_anomaly'] = combined['anomaly_score'] == -1

    return combined

# ── RUN PIPELINE ──────────────────────────────────────────────────
with st.spinner("Running pipeline..."):
    df = run_pipeline(n_transactions, contamination)

total_fraud = df['is_fraud'].sum()
caught = df[(df['is_fraud'] == 1) & (df['is_anomaly'] == 1)].shape[0]
false_alarms = df[(df['is_fraud'] == 0) & (df['is_anomaly'] == 1)].shape[0]
catch_rate = caught / total_fraud * 100

# ── METRICS ROW ───────────────────────────────────────────────────
st.subheader("📊 Pipeline Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Cases", total_fraud)
col3.metric("Anomalies Flagged", df['is_anomaly'].sum())
col4.metric("Fraud Caught", caught)
col5.metric("Catch Rate", f"{catch_rate:.1f}%")

st.divider()

# ── CHARTS ROW ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("💳 Batch vs Real-Time")
    payment_counts = df['payment_type'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(payment_counts, labels=payment_counts.index,
            autopct='%1.1f%%', colors=['#2196F3', '#4CAF50'])
    st.pyplot(fig1)

with col_right:
    st.subheader("📈 Transaction Amount Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df[df['is_fraud']==0]['amount'], bins=50,
             alpha=0.6, label='Normal', color='#2196F3')
    ax2.hist(df[df['is_fraud']==1]['amount'], bins=50,
             alpha=0.6, label='Fraud', color='#F44336')
    ax2.set_xlabel("Amount ($)")
    ax2.set_ylabel("Count")
    ax2.legend()
    st.pyplot(fig2)

st.divider()

# ── FRAUD DETECTION BREAKDOWN ─────────────────────────────────────
st.subheader("🚨 Fraud Detection Results")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Transaction Status")
    status_counts = df['status'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(status_counts.index, status_counts.values,
            color=['#4CAF50', '#FF9800', '#F44336'])
    ax3.set_ylabel("Count")
    st.pyplot(fig3)

with col_b:
    st.subheader("Anomalies by Hour of Day")
    anomalies = df[df['is_anomaly'] == True]
    hourly = anomalies.groupby('hour').size()
    fig4, ax4 = plt.subplots()
    ax4.bar(hourly.index, hourly.values, color='#9C27B0')
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Anomalies Flagged")
    st.pyplot(fig4)

st.divider()

# ── FLAGGED TRANSACTIONS TABLE ────────────────────────────────────
st.subheader("🔴 Flagged Transactions")
flagged = df[df['is_anomaly'] == True][
    ['transaction_id', 'sender', 'receiver', 'amount',
     'payment_type', 'status', 'timestamp', 'is_fraud']
].copy()
flagged['correctly_caught'] = flagged['is_fraud'] == 1
st.dataframe(flagged, use_container_width=True)

# ── RAW DATA ──────────────────────────────────────────────────────
if show_raw_data:
    st.subheader("📋 All Transactions")
    st.dataframe(df[['transaction_id', 'sender', 'receiver',
                      'amount', 'payment_type', 'status',
                      'timestamp', 'is_fraud', 'is_anomaly']],
                 use_container_width=True)

st.divider()
st.subheader("🎯 Model Performance Analysis")

col1, col2, col3 = st.columns(3)

cm = confusion_matrix(df["is_fraud"], df["is_anomaly"].astype(int))
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp) if (tp+fp) > 0 else 0
recall = tp/(tp+fn) if (tp+fn) > 0 else 0
f1 = 2*(precision * recall)/(precision + recall) if (precision + recall) > 0 else 0

col1.metric("Precision", f"{precision:.1%}",
    help="Of transactions flagged as fraud, how many actually were?")
col2.metric("Recall", f"{recall:.1%}",
    help="Of all actual fraud, how much did the model catch?")
col3.metric("F1 Score", f"{f1:.1%}",
    help="Balance between precision and recall")

st.info("""
**Why this matters for banking:**
- **High Recall** = catch more fraud but flag more legitimate payments (customer friction)
- **High Precision** = fewer false alarms but miss more fraud (financial loss)
- Banks typically prioritize Recall — missing fraud is more costly than a false alarm
""")

st.divider()
st.subheader("🔍 Fraud Detection by Type")

fraud_types = {
    'Late-night batch': df[df['transaction_id'].str.startswith('FRD')],
    'Structuring': df[df['transaction_id'].str.startswith('STR')],
    'Velocity fraud': df[df['transaction_id'].str.startswith('VEL')],
    'Ghost transactions': df[df['transaction_id'].str.startswith('GHT')],
    'Round numbers': df[df['transaction_id'].str.startswith('RND')],
}

fraud_summary = []
for fraud_type, subset in fraud_types.items():
    total = len(subset)
    caught = subset['is_anomaly'].sum()
    fraud_summary.append({
        'Fraud Type': fraud_type,
        'Total Injected': total,
        'Caught': int(caught),
        'Missed': int(total - caught),
        'Detection Rate': f"{caught/total*100:.0f}%" if total > 0 else "0%"
    })

fraud_summary_df = pd.DataFrame(fraud_summary)
st.dataframe(fraud_summary_df, use_container_width=True)

fig5, ax5 = plt.subplots(figsize=(10, 4))
x = range(len(fraud_summary_df))
ax5.bar(x, fraud_summary_df['Caught'], label='Caught', color='#4CAF50')
ax5.bar(x, fraud_summary_df['Missed'],
        bottom=fraud_summary_df['Caught'],
        label='Missed', color='#F44336')
ax5.set_xticks(x)
ax5.set_xticklabels(fraud_summary_df['Fraud Type'], rotation=15)
ax5.set_ylabel("Count")
ax5.set_title("Fraud Caught vs Missed by Type")
ax5.legend()
st.pyplot(fig5)

st.divider()
st.subheader("⚖️ Sensitivity Analysis")
st.caption("How does model sensitivity affect fraud detection vs false alarms?")

rates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
catch_rates_list = []
false_alarm_rates_list = []

features = ['amount', 'hour', 'day_of_week', 'is_weekend',
            'is_late_night', 'is_large_amount', 'amount_zscore',
            'payment_type_encoded', 'amount_vs_sender_avg',
            'sender_tx_count', 'is_round_number', 'is_tiny_amount']

for rate in rates:
    temp_model = IsolationForest(contamination=rate, random_state=42)
    preds = temp_model.fit_predict(df[features])
    is_anomaly_temp = preds == -1
    tp_temp = ((df['is_fraud'] == 1) & is_anomaly_temp).sum()
    fp_temp = ((df['is_fraud'] == 0) & is_anomaly_temp).sum()
    fn_temp = ((df['is_fraud'] == 1) & ~is_anomaly_temp).sum()
    catch_rates_list.append(tp_temp / (tp_temp + fn_temp) * 100 if (tp_temp + fn_temp) > 0 else 0)
    false_alarm_rates_list.append(fp_temp / len(df) * 100)

fig6, ax6 = plt.subplots(figsize=(10, 4))
ax6.plot(rates, catch_rates_list, 'g-o', label='Fraud Catch Rate %', linewidth=2)
ax6.plot(rates, false_alarm_rates_list, 'r-o', label='False Alarm Rate %', linewidth=2)
ax6.set_xlabel("Model Sensitivity (contamination rate)")
ax6.set_ylabel("Percentage")
ax6.legend()
ax6.set_title("Tradeoff: Catching More Fraud vs More False Alarms")
st.pyplot(fig6)

st.caption("""
**Key insight:** As sensitivity increases the model catches more fraud but also flags more 
legitimate transactions. The optimal point depends on the cost of missed fraud vs customer friction.
For institutions like FHLBC, missing a fraudulent batch payment is far more costly than a false alarm.
""")

ax5.bar(x, fraud_summary_df['Caught'], label='Caught', color='#1565C0')
ax5.bar(x, fraud_summary_df['Missed'], bottom=fraud_summary_df['Caught'],
        label='Missed', color='#FF6F00')

ax6.plot(rates, catch_rates_list, 'b-o', label='Fraud Catch Rate %', linewidth=2)
ax6.plot(rates, false_alarm_rates_list, color='#FF6F00', marker='o', label='False Alarm Rate %', linewidth=2)

ax1.pie(payment_counts, labels=payment_counts.index,
        autopct='%1.1f%%', colors=['#1565C0', '#00897B'])

ax2.hist(df[df['is_fraud']==0]['amount'], bins=50,
         alpha=0.6, label='Normal', color='#1565C0')
ax2.hist(df[df['is_fraud']==1]['amount'], bins=50,
         alpha=0.6, label='Fraud', color='#FF6F00')

st.markdown("""
<style>
.stMetric {
    background-color: #1E1E2E;
    border-radius: 8px;
    padding: 10px;
}
.stDataFrame {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)