import pandas as pd
df = pd.read_excel("FRPS_CY2024_IDR_data.xlsx", sheet_name=0)

print(df.head(20))
print("\nColumn names:")
print(df.columns.tolist())
print("\nSheet names available:")

x1 = pd.ExcelFile("FRPS_CY2024_IDR_data.xlsx")
print(x1.sheet_names)

import pandas as pd
df = pd.read_excel("FRPS_CY2024_IDR_data.xlsx", sheet_name="Table 1", header=None)
print(df.to_string())

df.dropna(how="all")
df = df.dropna(axis=1, how="all")

df = df.reset_index(drop=True)

print(df.to_string())

import pandas as pd
import numpy as np

ACH_AVG_AMOUNT = 2642
ACH_STD_DEV = 1500
FAILURE_RATE = 1.5
PENDING_RATE = 3.0
SUCCESS_RATE = 95.5

print("Federal Reserve ACH Statistics loaded:")
print(f"  Average transaction amount: ${ACH_AVG_AMOUNT:,}")
print(f"  Failure rate: {FAILURE_RATE}%")
print(f"  Success rate: {SUCCESS_RATE}%")