import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_transactions(n=100):
    transactions = []

    for i in range(n):
        payment_type = random.choice(["batch", "real-time"])
        amount = round(random.uniform(10,5000), 2)

        transaction = {
            "transaction_id": f"TXN{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": amount,
            "payment_type": payment_type,
            "status": random.choice(["completed", "pending", "failed"]),
            "timestamp": fake.date_time_this_month(),
        }
        transactions.append(transaction)

    df = pd.DataFrame(transactions)
    df.to_csv("transactions.csv", index=False)
    print(f"Generated {n} transactions!")
    print(df.head())
    return df
generate_transactions()

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
ACH_AVG_AMOUNT = 2642
ACH_STD_DEV = 1500
FAILURE_RATE = 1.5
PENDING_RATE = 3.0
SUCCESS_RATE = 95.5

def generate_transactions(n=100):
    transactions = []

    for i in range(n):
        payment_type = random.choices(
            ["batch", "real-time"],
            weights=[70,30]
        )[0]

        amount = round(np.random.normal(ACH_AVG_AMOUNT, ACH_STD_DEV), 2)
        amount = max(1, amount)

        status = random.choices(
            ["completed", "failed", "pending"],
            weights=[SUCCESS_RATE, FAILURE_RATE, PENDING_RATE]
        )[0]

        transaction = {
            "transaction_id": f"TXN{i+1:04d}",
            "sender": fake.name(),
            "receiver": fake.name(),
            "amount": amount,
            "payment_type": payment_type,
            "status": status,
            "timestamp": fake.date_time_this_month(),
        }
        transactions.append(transaction)

    df = pd.DataFrame(transactions)
    df.to_csv("transactions.csv", index=False)
    print(f"Generated {n} transactions!")
    print(df.head())
    print(f"\nPayment type breakdown:")
    print(df["payment_type"].value_counts())
    print(f"\nStatus breakdown:")
    print(df["status"].value_counts())
    return df

generate_transactions(n=1000)
