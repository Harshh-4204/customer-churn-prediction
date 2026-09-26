import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

df = pd.read_csv("customer_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

print(df.head())
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
print("\nChurn distribution:")
print(df["Churn"].value_counts())
churn_rate = (df["Churn"] == "Yes").mean() * 100

print(f"\nOverall churn rate: {churn_rate:.2f}%")
print("\nChurn rate by contract type:")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)
print("\nChurn rate by tenure group:")

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print(tenure_churn)
print("\nChurn rate by monthly charges:")

df["MonthlyChargesGroup"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 100, 150],
    labels=["0-35", "36-70", "71-100", "101+"]
)

charges_churn = pd.crosstab(
    df["MonthlyChargesGroup"],
    df["Churn"],
    normalize="index"
) * 100

print(charges_churn)
plt.figure(figsize=(8, 5))

bars = plt.bar(contract_churn.index, contract_churn["Yes"])

plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract Type")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom"
    )

plt.show()
plt.figure(figsize=(8, 5))

bars = plt.bar(tenure_churn.index.astype(str), tenure_churn["Yes"])

plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Customer Tenure")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom"
    )

plt.show()

plt.figure(figsize=(8, 5))

bars = plt.bar(
    charges_churn.index.astype(str),
    charges_churn["Yes"]
)

plt.xlabel("Monthly Charges")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Monthly Charges")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom"
    )

plt.show()

conn = sqlite3.connect("churn.db")

df.to_sql("customers", conn, if_exists="replace", index=False)

print("\nData loaded into SQLite database.")

query = """
SELECT
    Churn,
    COUNT(*) AS Customers,
    ROUND(AVG(MonthlyCharges), 2) AS AvgMonthlyCharges
FROM customers
GROUP BY Churn;
"""

result = pd.read_sql_query(query, conn)

print("\nMonthly charges by churn status:")
print(result)
conn.close()