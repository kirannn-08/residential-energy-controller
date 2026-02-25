import pandas as pd

df = pd.read_csv(
    "data/raw/household_power.csv",
    sep=",",
    low_memory=False
)

print(df.head())
print(df.info())
print(df.isna().sum())