import pandas as pd
import numpy as np


def main():

    df = pd.read_csv("data/processed/clean_household.csv")

    # Convert sub-metering Wh/min to kW
    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        df[col] = df[col] * 60 / 1000

    # Compute remainder load
    df["other_load"] = (
        df["Global_active_power"]
        - df["Sub_metering_1"]
        - df["Sub_metering_2"]
        - df["Sub_metering_3"]
    )

    # Remove negative rounding artifacts
    df["other_load"] = df["other_load"].clip(lower=0)

    df.to_csv("data/processed/ml_household.csv", index=False)

    print("ML dataset prepared.")
    print("Columns:", df.columns.tolist())
    print("Sample rows:")
    print(df.head())


if __name__ == "__main__":
    main()